import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',filename='exp/example.log')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

def main():

    logging.debug('这是一个调试消息')
    logging.info('这是一个信息消息')
    logging.warning('这是一个警告消息')
    logging.error('这是一个错误消息')
    logging.critical('这是一个严重错误消息')

if __name__ == "__main__":
    main()
