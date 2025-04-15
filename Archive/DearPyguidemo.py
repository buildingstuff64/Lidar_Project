import random
import string


def generate_random_string(length):
    # Generate a random string of a given length
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

print(generate_random_string(15))