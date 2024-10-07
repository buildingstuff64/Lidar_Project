import keyboard

class testclass:
    def __init__(self):
        self.x = 5
        self.b = 6
        self.c = 7

    def printallvariables(self):
        print(f"x={self.x}, b={self.b}, c={self.c}")


def printR():
    print("not q")

print("hello, world")
print("Ben Branch")

x = 5
b = [1, 2, 3, 4]
c = "string"

if (x > b[0]):
    print("x")

for x in range(0, 5):
    print(x)

while True:
    print("hello world")
    if keyboard.is_pressed('q'):
        break
    else:
        printR()

test = testclass()
test.printallvariables()


print("END OF PROGRAM")