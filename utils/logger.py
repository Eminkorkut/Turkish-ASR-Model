import logging
import sys
import os

def get_logger(name, log_file="train.log"):
    """
    Basit bir logger yapılandırması döndürür.
    Hem ekrana (konsol) hem de dosyaya log basar.
    
    Args:
        name (str): Logger ismi (genellikle __name__).
        log_file (str): Log dosyasının kaydedileceği yol.
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Eğer daha önce handler eklendiyse tekrar ekleme
    if not logger.handlers:
        # Format belirle
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 1. Konsol Handler (Ekrana basar)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 2. Dosya Handler (Dosyaya kaydeder)
        # Log dosyası ana dizine kaydedilsin
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
