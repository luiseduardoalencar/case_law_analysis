from loguru import logger
import sys
from pathlib import Path

# Adicionar src ao path se necessário
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from config.settings import LOG_CONFIG

# Remover handler padrão
logger.remove()

# Console handler
logger.add(
    sys.stdout,
    format="^<green^>{time:YYYY-MM-DD HH:mm:ss}^</green^> ^| ^<level^>{level: ^<8}^</level^> ^| ^<cyan^>{name}^</cyan^>:^<cyan^>{function}^</cyan^> - ^<level^>{message}^</level^>",
    level=LOG_CONFIG['level']
)

# File handler
logger.add(
    LOG_CONFIG['file'],
    format="{time:YYYY-MM-DD HH:mm:ss} ^| {level: ^<8} ^| {name}:{function}:{line} - {message}",
    level=LOG_CONFIG['level'],
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)
