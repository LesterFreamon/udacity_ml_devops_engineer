"""Calculate the total price of holiday gifts."""
import numpy as np


def open_file_and_convert_to_int(path: str) -> np.array:
    """Open file and convert to int."""
    with open(path, encoding='utf-8') as file:
        str_gift_costs = file.read().split('\n')
    converted_gift_costs = np.array(str_gift_costs).astype(int)
    return converted_gift_costs


def calculate_total_price(costs: np.array) -> float:
    """Calculate total price of gifts."""
    return (costs[costs < 25]).sum() * 1.08


if __name__ == "__main__":
    gift_costs = open_file_and_convert_to_int('gift_costs.py')
    total_price = calculate_total_price(gift_costs)
    print(total_price)
