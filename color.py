valid_hex = '0123456789ABCDEF'.__contains__
def cleanhex(data):
    return ''.join(filter(valid_hex, data.upper()))

def back_fromhex(text, hexcode):
    """print in a hex defined color"""
    hexint = int(cleanhex(hexcode), 16)
    print("\x1B[48;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text))

VAL = list(range(10)) + ['A', 'B', 'C', 'D', 'E', 'F']

for i in VAL:
  back_fromhex('value', f'{i}{i}0000')


print('This was', end=' ')
back_fromhex('a mistake', 'FF0000')