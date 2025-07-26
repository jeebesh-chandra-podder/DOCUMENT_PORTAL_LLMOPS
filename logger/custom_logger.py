import logging
import os
from datetime import datetime



class CustomLogger:
    def __init__(self, log_dir = "logs"):
        
        self.log_dir = log_dir
        logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(logs_dir, exist_ok=True)

        LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
        
        logging.basicConfig(
            filename=LOG_FILE_PATH,
            format="[ %(asctime)s ] [ %(levelname)s ]  [ %(name)s ] [ (line:%(lineno)d) ] - %(message)s",
            level=logging.DEBUG
        )

    def get_logger(self, name = __file__):
        return logging.getLogger(os.path.basename(name))


# Example usage:
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("Logging setup complete.")