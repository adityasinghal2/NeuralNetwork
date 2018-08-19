
def encode(key, string):
    encoded_string=""
    for j,i in enumerate(string):
        m=(ord(i)*key)
        encoded_string = encoded_string + "//" + str(m)
    return encoded_string
def decode(key, string):
    encoded_string = string.split("//")[1:]
    decoded_string = ""
    for j,i in enumerate(encoded_string):
        decoded_string = decoded_string+chr(int(i)*key)
    return decoded_string
print (encode(666, 'Attack at dawn!'))
print (decode(666, '//43290//77256//77256//64602//65934//71262//21312//64602//77256//21312//66600//64602//79254//73260//21978'))