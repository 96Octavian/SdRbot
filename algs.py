import random
from math import gcd, sqrt, floor
from typing import Generator, Dict


# TODO: Verify all conditions

def ee_info() -> str:
    text = "Find x and y such that ax + by = gcd(a, b)"
    return text


def ee_alg(a: int, b: int) -> (int, int):
    if a < b:
        a, b = b, a

    x0: int = 1
    y0: int = 0
    x1: int = 0
    y1: int = 1

    while True:
        q, r = divmod(a, b)
        x = x0 - (q * x1)
        y = y0 - (q * y1)
        x0 = x1
        y0 = y1
        x1 = x
        y1 = y
        if r == 0:
            break
        a = b
        b = r

    return x0, y0


def e_info() -> str:
    text = "Calculate gcd(a, b) using the Euclidean algorithm"
    return text


def e_alg(a: int, b: int) -> int:
    while True:
        _, r = divmod(a, b)
        if r == 0:
            break
        a = b
        b = r
    return b


def sm_info() -> str:
    text = "Calculate num^exp (mod p) using square&multiply\n"
    return text


def sm_alg(num: int, ex: int, p: int):
    exp = bin(ex)[2:]

    value: int = 1
    for i in range(0, len(exp)):
        value = value * value
        if exp[i] == '1':
            value = value * num
        value = value % p
    return value


def is_generator_info() -> str:
    text = ("Check if a is generator of Z*p\n"
            "For each prime factor of p - 1, a^(p-1/q_i) != 1 mod p"
            )
    return text


def is_generator_alg(generator: int, modulo: int) -> bool:
    for factor in prime_factors_alg(modulo - 1):
        if pow(generator, (modulo - 1) / factor, modulo) == 1:
            # TODO: Benchmark against math.pow(generator, modulo - 1) % modulo
            # TODO: Benchmark against sm(generator, modulo - 1, modulo)
            return False
    return True


def prime_factors_info() -> str:
    text = 'Prime factors of num\n/factor num'
    return text


def prime_factors_alg(n: int) -> Generator[int, None, None]:
    while not n & 1:
        yield 2
        n = n >> 1
    f = 3
    while f * f <= n:
        if n % f == 0:
            yield f
            n //= f
        else:
            f += 2
    if n != 1:
        yield n


def prime_factors_powered_info() -> str:
    text = 'Prime factors of num\n/factor num'
    return text


def prime_factors_powered_alg(num: int) -> Generator[int, None, None]:
    f = 1
    n: int = num
    while not n & 1:
        f = f << 1
        n = n >> 1

    if n != num:
        yield f

    f = 3
    while f * f <= n:
        tmp = divmod(n, f)
        if tmp[1] != 0:
            f += 2
        else:
            n = tmp[0]
            powered_factor = f
            tmp = divmod(n, f)
            while tmp[1] == 0:
                n = tmp[0]
                powered_factor *= f
                tmp = divmod(n, f)
            yield powered_factor
            f += 2

    if n != 1:
        yield n


def mod_info() -> str:
    text = ('Computes a mod b')
    return text


def mod_alg(a: int, b: int) -> int:
    return divmod(a, b)[1]


def verify_rsa_info() -> str:
    text = ("Verify:\n"
            "p and q must be prime\n"
            "exp must be relative prime with phi(p * q)"
            )
    return text


def verify_rsa_alg(p: int, q: int, exp: int) -> (bool, str):
    answer: str = ""
    response: bool = True
    factors = prime_factors_alg(p)
    first = next(factors, p)
    if p == first:
        answer += 'p is prime\n'
    else:
        response = False
        answer += f'p = {first} * {int(p / first)}\np is not prime\n'

    phi_n = (p - 1) * (q - 1)
    is_relative_prime = gcd(exp, phi_n) == 1
    answer += f'{exp} {"is" if is_relative_prime else "is not"} relative prime with phi(n) = {phi_n}\n'
    response = response or is_relative_prime
    return response, answer


def verify_gamal_info() -> str:
    text = ("Verify:\n"
            "p prime\n"
            "1 < a < p - 2\n"
            "alpha generator of Z*p (alpha^(p-1/q_i)!= 1 mod p)\n"
            "gcd(k, p - 1) = 1\n"
            )
    return text


def verify_gamal_alg(p: int, a: int, alpha: int, k: int = None) -> (bool, str):
    answer: str = ""
    response: bool = True
    p_factors = prime_factors_alg(p)
    first = next(p_factors, p)
    if p == first:
        answer += 'p is prime\n'
    else:
        response = False
        answer = f'p = {first} * {int(p / first)}\np is not prime\n'
    if a <= 1 or a >= p - 2:
        response = False
        answer += '1 < a < p - 2 is not true\n'
    else:
        answer += '1 < a < p - 2 is true\n'
    is_gen: bool = is_generator_alg(alpha, p)
    answer += f'{a} {"is" if is_gen else "is not"} a generator for p\n'
    response = response or is_gen
    if k:
        if gcd(k, p) != 1:
            response = False
            answer += "k and p are not relative primes"
        else:
            answer += 'k and p are relative primes\n'

    return response, answer


def coprimes_info() -> str:
    text = "Finds all the numbers up to n coprimes with n"
    return text


def coprimes_alg(n: int) -> Generator[int, None, None]:
    for k in range(1, n + 1):
        if gcd(n, k) == 1:
            yield k


def phi_info() -> str:
    text = ("Returns how many numbers up to n are relative primes to n\n"
            "phi(n) = product(factor(n) - 1)\n"
            )
    return text


def phi_alg(n: int) -> int:
    return sum(1 for _ in coprimes_alg(n))


def lambda_info() -> str:
    text = ("Let p_i be a factor of n and r its exponent, lambda(n) is equal to:\n"
            "lambda(n) = lcm({phi(p_i^r})\n"
            "If p^r is even, lambda(p^r) = phi(p^r)/2, else = phi(p^r)"
            )
    return text


def lambda_alg(num) -> int:
    powered_factors = prime_factors_powered_alg(num)
    phis = [len(list(coprimes_alg(x))) for x in powered_factors]
    mcm = phis[0]
    for elem in phis:
        mcm = lcm(mcm, elem)
    return mcm


def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def mr_info() -> str:
    text = ("Test if n is prime using Miller-Rabin method\n")
    return text


def mr(number: int, rounds: int = 40) -> bool:
    """Miller-Rabin primality test

    Test if number could be prime using the Miller-Rabin Primality Test with rounds rounds.
    A return value of false means number is definitely composite, while true means it is probably prime.
    The higher rounds is, the more accurate the test is.
    :param int number: The number to be tested for primality.
    :param int rounds: How many rounds to use in the test.
    :return: A bool indicating if the number could be prime or not.
    :rtype: bool

    """
    # Handle corner cases
    if number == 1:
        return False
    if number == 2:
        return True
    if number == 3:
        return True

    # Factor out the powers of 2 from {number - 1} and save the result
    d: int = number - 1
    r: int = 0
    while not d & 1:
        d = d >> 1
        r += 1

    # Cycle at most {round} times
    for _ in range(rounds + 1):
        a: int = random.randint(2, number - 2)
        x: int = pow(a, d, number)
        if x == 1 or x == number - 1:
            continue
        # Cycle at most {r - 1} times
        for _ in range(r):
            x = x * x % number
            if x == number - 1:
                break
        if x == number - 1:
            continue
        return False
    return True


def verify_bbs_info() -> str:
    text = ("p = q = 3 mod 4\n"
            "n = p * q\n"
            "gcd(seed, n) = 1\n"
            )
    return text


def verify_bbs_alg(p: int, q: int, seed: int) -> (bool, str):
    valid: bool = True

    mod_three: bool = p % 4 == 3
    valid &= mod_three
    text = f'p is{"" if mod_three else " not"} 3 mod 4\n'

    mod_three: bool = q % 4 == 3
    valid &= mod_three
    text += f'q is{"" if mod_three else " not"} 3 mod 4\n'

    n: int = q * p
    valid &= gcd(n, seed) == 1
    text += f'seed is{"" if valid else " not"} relative prime with n = p * q = {n}\n'

    return valid, text


def bbs_info() -> str:
    text = ("Blum-Blum-Shub pseudo random bit generator\n"
            "LSB is the randomly generated bit. Every x is the square of the preceding\n"
            "The period is one of the factors of lambda(lambda(n))\n"
            )
    return text


def bbs(p: int, q: int, seed: int) -> Generator[int, None, int]:
    n: int = q * p
    seed = pow(seed, 2, n)
    x: int = seed
    index = 0

    x = pow(x, 2, n)
    yield divmod(x, 2)[1]
    index += 1

    while x != seed:
        x = pow(x, 2, n)
        yield divmod(x, 2)[1]
        index += 1

    return index


# TODO: this sucks
def pollard_alg(n: int, a: int) -> (bool, int):
    d: int = 1
    b: int = a
    while d == 1:
        a = (a * a + 1) % n
        b = (b * b + 1) % n
        b = (b * b + 1) % n
        d = gcd(abs(a-b), n)

    return d != n, d


def pollard_info() -> str:
    text = ("Computes a non trivial factor of n\n"
            "Starting from b1 = a mod n, calculate b_j = b_(j-1) ** j mod n until j equals the limit. "
            "Now, if d = gcd(b - 1, n) is differente from 1, you have a non trivial factor of n\n"
            "/pollard n a limit\n")
    return text


def bsgs_info() -> str:
    text = ("a^x= b (mod p)\n"
            "Choose N so that N^2 >= p - 1\n"
            "list: a^j (mod p) for j in 0..N\n"
            "list: ba^(-Nk) for k in 0..N\n"
            "When an elem in second list is in first, x = j + Nk")
    return text


def bsgs_alg(base, target, mod, N=None) -> int:
    if not N:
        N = floor(sqrt(mod)) + 1
    baby: Dict[int, int] = {}
    val: int = 1
    baby[1] = 0
    for j in range(1, N):
        val = pow(base, j, mod)
        baby[val] = j

    _, a_inv = ee_alg(base, mod)

    a_inv_N = pow(a_inv, N, mod)

    val = target
    for k in range(0, N):
        print(f'{k}: {val}')
        if val in baby:
            return baby[val] + N * k
        val = val * a_inv_N % mod
