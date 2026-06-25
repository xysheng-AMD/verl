class SleepLevel:
    RELEASE_KV_CACHE_ONLY = 1
    RELEASE_ALL = 2


class IPCConfig:
    DEFAULT_BUCKET_SIZE_MB = 4096


class ATOMDefaults:
    SLEEP_LEVEL = SleepLevel.RELEASE_ALL
    TEMPERATURE = 1.0
    BATCH_TIMEOUT = 0.01
