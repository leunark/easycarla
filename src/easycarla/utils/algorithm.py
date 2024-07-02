import random

def select_percentage_threshold(my_list, percentage):
    threshold = percentage / 100
    return [element for element in my_list if random.random() < threshold]

