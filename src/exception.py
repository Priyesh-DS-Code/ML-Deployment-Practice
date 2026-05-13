import sys
from src.logger import logging

def error_message_details(error_message, error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msgg="The error is occured in file [{0}] line number [{1}] and error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error_message)
    )

    return error_msgg

class CustomException(Exception):
    def __init__(self, error_msg, error_detail):
        super().__init__(error_msg)
        self.error=error_message_details(error_message=error_msg, error_details=error_detail)

    def __str__(self):
        return self.error
    

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("divide by zero")
        raise CustomException(e, sys)