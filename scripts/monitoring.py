import logging

logging.basicConfig(level=logging.INFO, filename='model_performance.log')


def log_performance(user_id, response_time):
    logging.info(f"User ID: {user_id}, Response Time: {response_time:.2f}s")
