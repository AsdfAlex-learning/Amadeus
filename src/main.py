import sys
from pathlib import Path

from loguru import logger

from src.app.window import AmadeusWindow
from src.config import load_config


def main():
    config = load_config()

    logger.remove()
    logger.add(
        sys.stderr,
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    log_path = Path("logs/amadeus.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level=config["logging"]["level"],
        format=config["logging"]["format"],
        rotation=config["logging"]["rotation"],
        retention=config["logging"]["retention"],
    )

    logger.info(f"Starting {config['app']['name']} v{config['app']['version']}")

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName(config["app"]["name"])

    window = AmadeusWindow(config)
    window.show()

    logger.info("Application started successfully")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
