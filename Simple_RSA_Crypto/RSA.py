import random

def text2ints(text, m):
    st = text.encode()
    while(len(st) % m != 0):
        st += '\x00'.encode()
    a = [st[i:i+m] for i in range(0, len(st), m)]
    done_ar = []
    for s in a:
        done_ar.append(int.from_bytes(s, byteorder='big'))
    return done_ar


def ints2text(ints, m):
    int_ar = ints
    b_ar = []
    st_ar = []
    for i in int_ar:
        b_ar.append(i.to_bytes(m, byteorder='big'))
    for b in b_ar:
        st_ar.append(b.decode())


    str = "".join(st_ar)
    done = str.replace('\x00','')
    return done

def xgcd(a, b):
    q, x, y, x_last, y_last = 0, 0, 1, 1, 0
    while b != 0:
        q = a // b
        a, b = b, (a%b)
        x, x_last = (x_last - q * x), x
        y, y_last = (y_last - q * y), y
    return a, x_last, y_last


def gcd(a, b):
    while b != 0:
        a,b = b,(a % b)
    return a


def generate_keypair(p, q):
    n = p * q
    phi = (p-1) * (q-1)
    e = random.randrange(1,phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)

    d = xgcd(e, phi)[1]
    if e*d % phi == 1:
        #print("True")
        return ((e, n), (d, n)) if d > 0 else generate_keypair(p,q)
    else:
        return "Oof!"


def encrypt(pubkey, plaintext):
    e, n = pubkey
    m = 0
    while pow(2, 8*(m+1))-1 < seckey[1]:
        m += 1
    plain = text2ints(plaintext, m)
    chipher = [pow(char, e, n) for char in plain]
    return chipher

def decrypt(seckey, ciphertext):
    d, n = seckey
    m = 0
    while pow(2, 8*(m+1))-1 < seckey[1]:
        m += 1

    x = []
    for char in ciphertext:
        x.append(pow(char,d,n))
    return ints2text(x,m)

if __name__ == "__main__":
    p = int(input("Enter prime number p: "))
    q = int(input("Enter prime number q: "))
    print("Generating keypair")
    pubkey, seckey = generate_keypair(p, q)
    print("The public key is", pubkey, "and the private key is", seckey)
    message = input("Enter a message to encrypt with the public key: ")
    encrypted_msg = encrypt(pubkey, message)
    print("The encrypted message is: ")
    print(" ".join(map(lambda x: str(x), encrypted_msg)))
    print("The decrypted message is: ")
    print(decrypt(seckey, encrypted_msg))
