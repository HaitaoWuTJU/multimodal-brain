import logging

def set_logging(file_path):
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',filename=file_path,force=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)
