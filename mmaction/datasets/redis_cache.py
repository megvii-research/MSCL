import functools
import itertools
import json
import os
import pickle

import redis
import refile
from datamaid2.storage import OSS
from meghair.utils import io, logconf
from tqdm import tqdm

_REDIS_URL = os.environ.get(
    "VIDEO3_REDIS_URL", "redis://reid-redis-proxy.group-datamaid_admin.ws2.hh-b.brainpp.cn:6379/1",
)
_REDIS_MASTER_URL = os.environ.get(
    "VIDEO3_REDIS_MASTER_URL", "redis://reid-infrastructure.group-datamaid_admin.ws2.hh-b.brainpp.cn:6379/1",
)

logger = logconf.get_logger(__name__)


class RedisCachedIterator:

    _data_prefix = "rci"
    _ttl = 60 * 60 * 24 * 30  # 30 days

    def __init__(self, prefix, redis_url=None, redis_master_url=None, **kwargs):
        self.prefix = ".".join([self._data_prefix, prefix])
        self.redis_url = _REDIS_URL if redis_url is None else redis_url
        self.redis_master_url = _REDIS_MASTER_URL if redis_master_url is None else redis_master_url
        # logger.info("redis url is " + self.redis_url)
        if self._exist() and self._should_update_ttl():
            self._update_ttl()

    @functools.lru_cache(maxsize=2)
    def get_client(self, master=False):
        if master:
            return redis.Redis.from_url(self.redis_master_url)
        else:
            return redis.Redis.from_url(self.redis_url)

    @property
    def _client(self):
        return self.get_client(master=False)

    @property
    def _master_client(self):
        return self.get_client(master=True)

    @property
    def ttl(self):
        return self._client.ttl("{}.total".format(self.prefix))

    def _update_ttl(self):
        pipe = self._master_client.pipeline()
        for idx in range(len(self)):
            key = "{}.{}".format(self.prefix, idx)
            pipe.expire(key, self._ttl)
            if idx and idx % 1000 == 0:
                pipe.execute()
                pipe = self._master_client.pipeline()
        key = "{}.total".format(self.prefix)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def _should_update_ttl(self):
        if self._ttl:
            current_ttl = self._client.ttl("{}.total".format(self.prefix))
            current_ttl = current_ttl if current_ttl else 0
            # only update expire when current_ttl is less than 0.9 * target_ttl
            # in order to reduce the load of matser redis server
            result = current_ttl < self._ttl * 0.9
            if result:
                logger.info("[TTL] prefix: {} update ttl {} -> {}".format(self.prefix, current_ttl, self._ttl))
            return result
        else:
            return False

    def __len__(self):
        key = "{}.total".format(self.prefix)
        return json.loads(self._client.get(key).decode())

    def __get_doc(self, doc):
        assert isinstance(doc, str) and doc.startswith(self._data_prefix)
        return json.loads(self._client.get(doc).decode())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = []
            for i in itertools.islice(range(len(self)), idx.start, idx.stop, idx.step):
                key = "{}.{}".format(self.prefix, i)
                items.append(self.__get_doc(key))
            return items
        else:
            key = "{}.{}".format(self.prefix, idx)
            return self.__get_doc(key)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def _exist(self):
        key = "{}.total".format(self.prefix)
        return self._client.exists(key)

    def data_gen(self):
        raise NotImplementedError

    def _init_redis(self, rebuild=False, chunk_size=1000, **kwargs):
        pipe = self._master_client.pipeline()
        total = 0
        for idx, item in enumerate(self.data_gen()):
            key = "{}.{}".format(self.prefix, idx)
            pipe.set(key, json.dumps(item), ex=self._ttl)
            total += 1
            if idx % chunk_size == 0:
                pipe.execute()
                pipe = self._master_client.pipeline()
        key = "{}.total".format(self.prefix)
        pipe.set(key, json.dumps(total), ex=self._ttl)
        pipe.execute()


class RedisCachedPickle(RedisCachedIterator):
    """ cache a pickle on oss in redis

    :param path: path to the pickle file (oss only)
    :type mode: str
    :param ttl: ttl time, unit: second
    :type mode: int
    """

    def __init__(self, path, ttl=60 * 60 * 24, rebuild=False, redis_url=None, redis_master_url=None, **kwargs):
        oss = OSS()
        etag = oss.get_etag(path)
        self.path = path
        self._ttl = ttl
        super().__init__(prefix="opl.{}".format(etag), redis_url=redis_url, redis_master_url=redis_master_url, **kwargs)
        if not self._exist() or rebuild:
            logger.info("cannot find cache, building...")
            logger.info("cache path: {}".format(self.path))
            logger.info("cache key: {}".format(etag))
            self._init_redis()

    def data_gen(self):
        with refile.smart_open(self.path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, (list, tuple)), "only list/tuple is supported"
        yield from tqdm(data, disable=None)


class ActivityRedisCachedPickle(RedisCachedPickle):
    def data_gen(self):
        with refile.smart_open(self.path, "rb") as f:
            data = io.load(f)
            data = self.dataset_transformer(data)
        assert isinstance(data, (list, tuple)), "only list/tuple is supported"
        yield from tqdm(data, disable=None)

    def dataset_transformer(self, data):
        new_data = []
        if isinstance(data, dict):
            for pid, anno in data.items():
                anno["pid"] = pid
                if "snapshot_index" in anno.keys():
                    anno["snapshot_index"] = int(anno["snapshot_index"])
                new_data.append(anno)
        elif isinstance(data, list):
            new_data = data
            # for anno in data:
            #     if "snapshot_index" in anno.keys():
            #         anno["snapshot_index"] = int(anno["snapshot_index"])
            #     new_data.append(anno)
        else:
            raise NotImplementedError("not supported data type: {}".format(type(data)))

        return new_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("action", choices=["create", "info"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.action == "create":
        cache = RedisCachedPickle(args.path, rebuild=args.rebuild)

    elif args.action == "info":
        cache = RedisCachedPickle(args.path)
        logger.info("path: {}".format(args.path))
        logger.info("cache_key: {}".format(cache.prefix))
        logger.info("length: {}".format(len(cache)))
        logger.info("ttl: {}".format(cache.ttl))
