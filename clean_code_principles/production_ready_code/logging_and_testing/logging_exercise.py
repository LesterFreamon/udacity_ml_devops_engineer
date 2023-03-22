"""Logging Exercise."""
import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
    )


def sum_vals(first_number: int, second_number: int) -> int:
    """Sum two values.
    
    Args:
        first_number: (int) first number to sum
        second_number: (int) second number to sum

    Returns:
        (int) sum of first_number and second_number
    """
    try:
        assert isinstance(first_number, int) and isinstance(second_number, int)
        logging.info('SUCCESS: first_number and second_number are ints')
    except AssertionError:
        logging.error('ERROR: first_number and second_number must be ints')
    return first_number+second_number

if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
