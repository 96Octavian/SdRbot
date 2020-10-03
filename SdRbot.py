import random
import sys
from math import gcd, sqrt, floor
from typing import List

from telegram.ext import CommandHandler, Updater, BaseFilter


class FilterNoArgs(BaseFilter):
    def filter(self, message):
        return len(message.text.split(' ')) == 1


class FilterWithArgs(BaseFilter):
    def filter(self, message):
        return len(message.text.split(' ')) > 1


def verifyg_info(update, context):
    text = ("/verifyg p a alpha [k]\n"
            "hypothesis:\n"
            "p prime\n"
            "1 < a < p - 2\n"
            "alpha generator of Z*p (alpha^(p-1/q_i)!= 1 mod p)\n"
            "gcd(k, p - 1) = 1\n"
            )
    update.message.reply_text(text)


def verifyg(update, context):
    args = update.message.text.split(' ')
    del args[0]
    args = [int(x) for x in args]
    valid, text = verifyg_alg(*args)
    if valid:
        text += "\nParameters valid"
    else:
        text += "\nParameters not valid"
    update.message.reply_text(text)


def verifyg_alg(p: int, a: int, alpha: int, k: int = None) -> (bool, str):
    answer = ""
    p_factors = prime_factors(p)
    if p in p_factors:
        answer += 'p is prime\n'
    else:
        return False, f'p = {p_factors}\np is not prime'
    if a <= 1 or a >= p - 2:
        return False, answer + '1 < a < p - 2 is not true'
    else:
        answer += '1 < a < p - 2 is true\n'
    is_gen, text = is_generator_alg(alpha, p)
    answer += text
    if not is_gen:
        return False, answer
    if k:
        if gcd(k, p) != 1:
            return False, answer + "k and p are not relative primes"
        else:
            return True, answer + 'k and p are relative primes\n'
    else:
        return True, answer


def el_gamal(update, context):
    # TODO: Without message
    # TODO: Inverse
    command, p, a, k, alpha, message = update.message.text.split(' ')
    p = int(p)
    a = int(a)
    k = int(k)
    alpha = int(alpha)
    message = int(message)

    sm_text, beta = sm_alg(alpha, a, p)
    text = f'\nbeta = alpha^a mod p = {beta}\n{sm_text}\n'

    if k == 0:
        update.message.reply_text(text)
        return

    sm_text, r = sm_alg(alpha, k, p)
    text += f'\nr = alpha^k mod p = {r}\n{sm_text}\n'

    x, inverse_k, ee_text = ee_alg(p - 1, k)
    text += '\n' + ee_text + '\n'
    text += (f's = k^-1 (P-ar) mod (p - 1) = '
             '{inverse_k} ({message} - {a} * {r}) mod ({p} - 1) = '
             '{(inverse_k * (message - a * r)) % (p - 1)}\n')

    sm_text, beta_k = sm_alg(beta, k, p)
    text += f'\nbeta^k = {beta_k}\n{sm_text}\n'
    text += f't = beta^k P mod p = {beta_k} * {message} mod {p} = {(beta_k * message) % p}'
    update.message.reply_text(text)


def el_gamal_info(update, context):
    text = ("/el_gamal p a k alpha m\n"
            "hypothesis:\n"
            "p prime\n"
            "1 < a < p - 2\n"
            "alpha generator of Z*p (alpha^(p-1/q_i)!= 1 mod p)\n"
            "k _|_ p - 1\n"
            "Calculation:\n"
            "beta = alpha^a mod p\n"
            "r = alpha^k mod p\n"
            "t = beta^k m mod p\n"
            "C = (r, t)\n"
            "m = t r^-a = t r^(p-1-a)\n"
            "s = k^-1 (m-ar) mod (p - 1)\n"
            "Sig = (r, s)\n"
            "Verification:\n"
            "beta^r r^s = alpha^m mod p\n"
            "Nonce: (r, s1), (r, s2)\n"
            "(s1 - s2)k = m1 - m2 (mod (p - 1))\n"
            "Nonce: (r, t1), (r, t2)\n"
            "m2 = (t2/t1)m1 mod p"
            )
    update.message.reply_text(text)


def mod_info(update, context):
    text = ('/mod a b\n'
            'Computes a mod b')
    update.message.reply_text(text)


def mod(update, context):
    command, a, b = update.message.text.split(' ')
    a = int(a)
    b = int(b)
    update.message.reply_text(a % b)


def prime_factors(n: int) -> List[int]:
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def factor_info(update, context):
    text = 'Prime factors of num\n/factor num'
    update.message.reply_text(text)


def factor(update, context):
    command, number = update.message.text.split(' ')
    update.message.reply_text(prime_factors(int(number)))


def lambda_info(update, context):
    text = ("Let p_i be a factor of n and r its exponent, lambda(n) is equal to:\n"
            "lambda(n) = lcm({phi(p_i^r})\n"
            "If p^r is even, lambda(p^r) = phi(p^r)/2, else = phi(p^r)"
            )
    update.message.reply_text(text)


def lambda_alg(num):
    factors = prime_factors(int(num))
    powered_factors = set([x ** factors.count(x) for x in factors])
    phis = [len(phi_alg(x)) for x in powered_factors]
    mcm = phis[0]
    for elem in phis:
        mcm = lcm(mcm, elem)
    return factors, phis, mcm


def lambdaa(update, context):
    command, num = update.message.text.split(' ')
    num = int(num)

    factors, phis, mcm = lambda_alg(num)
    powered_factors = set([x ** factors.count(x) for x in factors])
    addendum = ""
    if len(powered_factors) != len(factors):
        addendum = f' = {powered_factors}'
    text = f'factors: {factors}{addendum}\nphis: {phis}\nlcm: {mcm}'

    update.message.reply_text(text)


def phi_info(update, context):
    text = ("/phi n\n"
            "Returns how many numbers up to n are relative primes to n\n"
            "phi(n) = produttoria(factor(n) - 1)\n"
            )
    update.message.reply_text(text)


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def phi_alg(n):
    nums = []
    for k in range(1, n + 1):
        if gcd(n, k) == 1:
            nums.append(k)
    return nums


def phi(update, context):
    command, n = update.message.text.split(' ')

    nums = phi_alg(int(n))

    update.message.reply_text(f'phi({n}) = {len(nums)}:\n{nums}')


def bbs_info(update, context):
    text = ("/bbs p q x\n"
            "Blum-Blum-Shub pseudo random bit generator\n"
            "p = q = 3 mod 4\n"
            "n = p * q\n"
            "gcd(x, n) = 1\n"
            "LSB is the randomly generated bit. Every x is the square of the preceding\n"
            "The period is one of the factors of lambda(lambda(n))\n"
            )
    update.message.reply_text(text)


def dividers(num):
    num = int(num)
    nums = []
    for i in range(1, int(num / 2) + 1):
        if num % i == 0:
            nums.append(i)
    nums.append(num)
    return nums


def bbs(update, context):
    command, p, q, x = update.message.text.split(' ')
    p = int(p)
    q = int(q)
    x = int(x)
    n = q * p
    init = (x ** 2) % n
    index = -1
    text = f'n = p * q = {n}\n'

    factors, phis, mcm = lambda_alg(n)
    powered_factors = set([x ** factors.count(x) for x in factors])
    addendum = ""
    if len(powered_factors) != len(factors):
        addendum = f' = {powered_factors}'
    text += f'factors: {factors}{addendum}\nphis: {phis}\nlcm: {mcm}\n'
    factors, phis, mcm = lambda_alg(mcm)
    powered_factors = set([x ** factors.count(x) for x in factors])
    addendum = ""
    if len(powered_factors) != len(factors):
        addendum = f' = {powered_factors}'
    text += f'factors: {factors}{addendum}\nphis: {phis}\nlcm: {mcm}\nlambda(lambda(n)): {mcm}\n'
    text += f'Period is one of {dividers(mcm)}\n'

    current = x
    while True:
        current = (current ** 2) % n
        index += 1
        text += f'X_{index} = {current}\n'
        if current == init and index != 0:
            break

    text += f'Period is {index}'
    update.message.reply_text(text)


def is_generator_info(update, context):
    text = ("/is_generator a p\n"
            "a generator of Z*p\n"
            "For each prime factor of p - 1, a^(p-1/q_i) != 1 mod p"
            )
    update.message.reply_text(text)


def is_generator_alg(a: int, p: int) -> (bool, str):
    factors = prime_factors(p - 1)
    text = f'Factors of p - 1: {factors}\n'
    is_gen = True
    for elem in set(factors):
        res = (a ** int((p - 1) / elem)) % p
        is_gen &= res != 1
        text += f'{a}^{int((p - 1) / elem)} = {res} mod {p} {"=" if res == 1 else "!="} 1 mod {p}\n'
    text += f'{a} {"is" if is_gen else "is not"} a generator for p\n'
    return is_gen, text


def is_generator(update, context):
    args = update.message.text.split(' ')
    del args[0]
    args = [int(x) for x in args]
    answer = is_generator_alg(*args)[1]
    update.message.reply_text(answer)


def e_info(update, context):
    text = ("/e a b\n"
            "Calculate gcd(a, b) using the Euclidean algorithm"
            )
    update.message.reply_text(text)


def e_alg(a, b):
    results = []
    while True:
        q = int(a / b)
        r = a % b
        results.append((a, b, q, r))
        if r == 0:
            break
        a = b
        b = r
    return results


def e(update, context):
    command, a, b = update.message.text.split(' ')
    a = int(a)
    b = int(b)
    if a < b:
        a, b = b, a
    a_orig = a
    b_orig = b

    text = f'gcd({a_orig}, {b_orig}):\n'

    results = e_alg(a, b)
    for (a, b, q, r) in results:
        text += f'{a} = {q} * {b} + {r}\n'
    text += f'gcd({a_orig}, {b_orig}) = {results[-1][1]}'

    update.message.reply_text(text)


def ee_info(update, context):
    text = ("/ee a b\n"
            "find x and y in ax + by = gcd(a, b)"
            )
    update.message.reply_text(text)


def ee_alg(a, b):
    if a < b:
        a, b = b, a
    a_orig = a
    b_orig = b

    text = f'gcd({a_orig}, {b_orig}):\n'

    results = e_alg(a_orig, b_orig)
    for (a, b, q, r) in results:
        text += f'{a} = {q} * {b} + {r}\n'

    text += f'{a_orig} x + {b_orig} y = gcd({a_orig}, {b_orig}) = {results[-1][1]}\n'

    x0 = 1
    y0 = 0
    x1 = 0
    y1 = 1
    index = 0
    text_x = f'X_{index} = {x0}\n'
    text_y = f'Y_{index} = {y0}\n'
    index += 1
    text_x += f'X_{index} = {x1}\n'
    text_y += f'Y_{index} = {y1}\n'
    index += 1
    del results[-1]
    x = 0
    y = 0
    for (a, b, q, r) in results:
        x = x0 - (q * x1)
        y = y0 - (q * y1)
        text_x += f'X_{index} = {x}\n'
        text_y += f'Y_{index} = {y}\n'
        index += 1
        x0 = x1
        y0 = y1
        x1 = x
        y1 = y
    text += f'\n{text_x}\n{text_y}\n'
    return x, y, text


def ee(update, context):
    command, a, b = update.message.text.split(' ')
    a = int(a)
    b = int(b)

    x, y, text = ee_alg(a, b)

    update.message.reply_text(text)


# TODO: Chinese remainder theorem


def rsa_info(update, context):
    text = ("/rsa p q e [message]\n"
            "Given two primes, their product n is the module\n"
            "e must be relative prime with phi(n)\n"
            "(n, e) is public\n"
            "encryption is c = m^e(mod n)\n"
            "Decryption uses the inverse of e (mod n)")
    update.message.reply_text(text)


def rsa_alg(p, q, exp, message=None):
    p = int(p)
    q = int(q)
    n = p * q
    exp = int(exp)
    message = int(message)
    phi_n = (p - 1) * (q - 1)
    if gcd(exp, phi_n) != 1:
        return "exp is not relative prime with phi(n)"
    x, y, text = ee_alg(phi_n, exp)
    d = y % phi_n
    text += f'd = e^-1(mod phi(n)) = {d}\n'
    text += f'Public key is ({n}, {exp})\n'
    if message != 0:
        text += f'c = m^e mod n = {(message ** exp) % n}'

    return text


def rsa(update, context):
    args = update.message.text.split(' ')
    del args[0]
    args = [int(x) for x in args]
    text = rsa_alg(*args)
    update.message.reply_text(text)


def mr_info(update, context):
    text = ("/mr n k [a]\n"
            "Test if n is prime\n"
            "Write n-1 as 2^r * m, then calculate b = a^m mod n: if = +-1 it may be prime\n"
            "Else, calculate b^2 mod n: if = 1 it's composite and gcd(b-1,n) is a factor, "
            "if -1 it may be prime, else continue\n"
            "Repeat k times")
    update.message.reply_text(text)


def mr(update, context):
    args = update.message.text.split(' ')
    del args[0]
    args = [int(x) for x in args]
    text = mr_alg(*args)
    update.message.reply_text(text)


def mr_alg(n, k, a=None):
    m = n - 1
    while m % 2 == 0:
        m /= 2
    m = int(m)
    text = f'm = {m}\n'
    if not a:
        a = random.randint(2, n - 2)
    text += f'a = {a}\n'
    b = (a ** m) % n
    text += f'b_{0} = {b}\n'
    if b == 1 or b == n - 1:
        text += f'{n} may be prime\n'
        return text
    for i in reversed(range(1, k)):
        b = (b ** 2) % n
        text += f'b_{k - i} = {b}\n'
        if b == 1:
            text += f'{n} is not prime\n'
            break
        if b == n - 1:
            text += f'{n} may be prime\n'
            break
    return text


def bsgs_info(update, context):
    text = ("/bsgs a b p [N]\n"
            "a^x= b (mod p)\n"
            "Choose N so that N^2 >= p - 1\n"
            "list: a^j (mod p) for j in 0..N\n"
            "list: ba^(-Nk) for k in 0..N\n"
            "When an elem in second list is in first, x = j + Nk")
    update.message.reply_text(text)


def bsgs(update, context):
    args = update.message.text.split(' ')
    del args[0]
    args = [int(x) for x in args]
    text = mr_alg(*args)
    update.message.reply_text(text)


def bsgs_alg(a, b, p, N=None):
    if not N:
        N = floor(sqrt(p - 1)) + 1
    text = f'N = {N}\n'
    baby = {}
    text += "Baby steps list:\n"
    val = 1 / a
    for j in range(0, N):
        val = int((val * a) % p)
        baby[val] = j
        text += f'{j}: {val}\n'

    x, a_inv, dump = ee_alg(a, p)

    text += "Giant steps list:\n"
    a_inv_N = (a_inv ** N) % p
    text += f'a^1 = {a_inv}\na^-N = {a_inv_N}\n'
    val = b
    text += f'0: {val}\n'
    if val in baby:
        text += f'x = j + Nk = {baby[val]} + {N} * {0} = {baby[val] + N * 0}\n'
        return text
    a_inv_N_k = 1
    for k in range(1, N):
        a_inv_N_k *= a_inv_N
        val = (b * a_inv_N_k) % p
        text += f'{k}: {val}\n'
        if val in baby:
            text += f'x = j + Nk = {baby[val]} + {N}*{k} = {baby[val] + N * k}\n'
            break
    return text


def sm_info(update, context):
    text = "/sm num exp p\n"
    update.message.reply_text(text)


def sm_alg(num, ex, p):
    exp = bin(int(ex))[2:]
    num = int(num)
    p = int(p)

    text = f'x = num^exp (mod p) = {num}^{ex} (mod {p})\n'
    text += f'bin(exp)={exp}\n'

    value = 1
    for i in range(0, len(exp)):
        text += f'{exp[i]}: {value}**2'
        value = value * value
        if exp[i] == '1':
            text += f' * {num}'
            value = value * num
        text += f' = {value}'
        value = value % p
        text += f' = {value} (mod {p})\n'
    return text, value


def sm(update, context):
    command, num, ex, p = update.message.text.split(' ')

    text, value = sm_alg(num, ex, p)

    update.message.reply_text(text)


def pollard_alg(n, a, limit):
    n = int(n)
    a = int(a)
    limit = int(limit)
    sequence = []
    b = a % n
    for i in range(2, limit + 1):
        b = (b ** i) % n
        sequence.append(b)
    return sequence, gcd(sequence[-1] - 1, n)


def pollard(update, context):
    command, n, a, limit = update.message.text.split(' ')
    sequence, d = pollard_alg(n, a, limit)
    text = f'b1 = a (mod n) = {a}\n'
    text += "b_j = b_(j-1) ** j\n"
    index = 2
    for elem in sequence:
        text += f'b{index} = {elem}\n'
        index += 1
    text += f'd = gcd(b - 1, n) = {d}\n'
    update.message.reply_text(text)


def pollard_info(update, context):
    text = ("Computes a non trivial factor of n\n"
            "Starting from b1 = a mod n, calculate b_j = b_(j-1) ** j mod n until j equals the limit. "
            "Now, if d = gcd(b - 1, n) is differente from 1, you have a non trivial factor of n\n"
            "/pollard n a limit\n")
    update.message.reply_text(text)


def command_help(update, context):
    text = ("/verifyg\n"
            "/elgamal\n"
            "/mod\n"
            "/factor\n"
            "/lambda\n"
            "/phi\n"
            "/bbs\n"
            "/is_generator\n"
            "/e\n"
            "/ee\n"
            "/rsa\n"
            "/mr\n"
            "/bsgs\n"
            "/sm\n"
            "/pollard\n")
    update.message.reply_text(text)


# Log errors
def error_logger(update, context):
    """Log Errors caused by Updates."""
    text = f'Error "{context.error}"'
    print(text)
    update.message.reply_text(text)


def main():
    # Create the Updater and pass it your bot's token.
    updater = Updater(sys.argv[1], use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # dp.add_handler(CallbackQueryHandler(log_on_callback_query), group=LOGGING)
    # dp.add_handler(MessageHandler(Filters.all, log_on_chat_message), group=LOGGING)

    # All messages
    # dp.add_handler(MessageHandler(Filters.all, contacts_logger), group=PREPROCESS)

    # On different commands
    # ~
    filter_no_args = FilterNoArgs()
    dp.add_handler(CommandHandler("verifyg", verifyg_info, filter_no_args))
    dp.add_handler(CommandHandler("verifyg", verifyg, ~filter_no_args))
    dp.add_handler(CommandHandler("v", verifyg_info, filter_no_args))
    dp.add_handler(CommandHandler("v", verifyg, ~filter_no_args))
    dp.add_handler(CommandHandler("el_gamal", el_gamal_info, filter_no_args))
    dp.add_handler(CommandHandler("el_gamal", el_gamal, ~filter_no_args))
    dp.add_handler(CommandHandler("g", el_gamal_info, filter_no_args))
    dp.add_handler(CommandHandler("g", el_gamal, ~filter_no_args))
    dp.add_handler(CommandHandler("mod", mod_info, filter_no_args))
    dp.add_handler(CommandHandler("mod", mod, ~filter_no_args))
    dp.add_handler(CommandHandler("factor", factor_info, filter_no_args))
    dp.add_handler(CommandHandler("factor", factor, ~filter_no_args))
    dp.add_handler(CommandHandler("f", factor_info, filter_no_args))
    dp.add_handler(CommandHandler("f", factor, ~filter_no_args))
    dp.add_handler(CommandHandler("lambda", lambda_info, filter_no_args))
    dp.add_handler(CommandHandler("lambda", lambdaa, ~filter_no_args))
    dp.add_handler(CommandHandler("l", lambda_info, filter_no_args))
    dp.add_handler(CommandHandler("l", lambdaa, ~filter_no_args))
    dp.add_handler(CommandHandler("phi", phi_info, filter_no_args))
    dp.add_handler(CommandHandler("phi", phi, ~filter_no_args))
    dp.add_handler(CommandHandler("bbs", bbs_info, filter_no_args))
    dp.add_handler(CommandHandler("bbs", bbs, ~filter_no_args))
    dp.add_handler(CommandHandler("is_generator", is_generator_info, filter_no_args))
    dp.add_handler(CommandHandler("is_generator", is_generator, ~filter_no_args))
    dp.add_handler(CommandHandler("ig", is_generator_info, filter_no_args))
    dp.add_handler(CommandHandler("ig", is_generator, ~filter_no_args))
    dp.add_handler(CommandHandler("e", e_info, filter_no_args))
    dp.add_handler(CommandHandler("e", e, ~filter_no_args))
    dp.add_handler(CommandHandler("ee", ee_info, filter_no_args))
    dp.add_handler(CommandHandler("ee", ee, ~filter_no_args))
    dp.add_handler(CommandHandler("rsa", rsa_info, filter_no_args))
    dp.add_handler(CommandHandler("rsa", rsa, ~filter_no_args))
    dp.add_handler(CommandHandler("mr", mr_info, filter_no_args))
    dp.add_handler(CommandHandler("mr", mr, ~filter_no_args))
    dp.add_handler(CommandHandler("bsgs", bsgs_info, filter_no_args))
    dp.add_handler(CommandHandler("bsgs", bsgs, ~filter_no_args))
    dp.add_handler(CommandHandler("sm", sm_info, filter_no_args))
    dp.add_handler(CommandHandler("sm", sm, ~filter_no_args))
    dp.add_handler(CommandHandler("pollard", pollard_info, filter_no_args))
    dp.add_handler(CommandHandler("pollard", pollard, ~filter_no_args))
    dp.add_handler(CommandHandler("p", pollard_info, filter_no_args))
    dp.add_handler(CommandHandler("p", pollard, ~filter_no_args))

    dp.add_handler(CommandHandler("help", command_help))

    # log all errors
    dp.add_error_handler(error_logger)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

    return


if __name__ == '__main__':
    main()
