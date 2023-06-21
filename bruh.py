class A:
    x: int
    l: list = []


class B(A):
    x = 2

    def __init__(self) -> None:
        super().__init__()
        self.l.append(self.x)


class C(A):
    x = 3

    def __init__(self) -> None:
        super().__init__()
        self.l.append(self.x)


class D(B, C):
    x = 4

    def __init__(self) -> None:
        super().__init__()
        self.l.append(self.x)


def main():
    # a1 = B()
    # print(a1.l)
    # a2 = C()
    # print(a1.l, a2.l)
    # a3 = D()
    # print(a1.l, a2.l, a3.l)
    a1 = B()
    print(a1.x)
    print(A.x, B.x)
    a2 = C()
    print(a1.x, a2.x)
    print(A.x, B.x, C.x)
    a3 = D()
    print(a1.x, a2.x, a3.x)
    print(A.x, B.x, C.x, D.x)


if __name__ == "__main__":
    main()
