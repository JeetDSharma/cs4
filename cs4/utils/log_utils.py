import logging, logging.config, os, pathlib, yaml

def setup_logging(cfg_path: str, job_log_file: str | None = None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ensure directories exist
    for handler in cfg.get("handlers", {}).values():
        if "filename" in handler:
            pathlib.Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

    # if a job-specific log file is provided, clone the file handler
    if job_log_file:
        job_dir = pathlib.Path(job_log_file).parent
        job_dir.mkdir(parents=True, exist_ok=True)
        cfg["handlers"]["job_file"] = {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "std",
            "filename": str(job_log_file),
        }
        # attach job_file handler to CS4 loggers
        for name in ["CS4Downloader", "CS4Parser", "CS4Generator", "CS4Evaluator"]:
            if name in cfg.get("loggers", {}):
                cfg["loggers"][name]["handlers"].append("job_file")

    logging.config.dictConfig(cfg)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
