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
    a1 = B()
    print(a1.l)
    a2 = C()
    print(a1.l, a2.l)
    a3 = D()
    print(a1.l, a2.l, a3.l)


if __name__ == "__main__":
    main()
