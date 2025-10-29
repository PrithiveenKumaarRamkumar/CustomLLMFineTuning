"""PII-like phone number for testing sanitization.
Phone: +1-415-555-2671
"""

from typing import List


def fibonacci(n: int) -> List[int]:
    a, b = 0, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq

if __name__ == "__main__":
    print(fibonacci(10))
