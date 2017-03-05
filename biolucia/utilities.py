class lazy_property:
    def __init__(self, f):
        self.f = f
        self.f_name = f.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.f(obj)
        setattr(obj, self.f_name, value)
        return value


def initialized_property(initializer):
    class _initialized_property:
        def __init__(self, f):
            self.f = f
            self.f_name = f.__name__

        def __get__(self, obj, cls):
            if obj is None:
                return self
            initializer(obj)
            return getattr(obj, self.f_name)

    return _initialized_property


class run_once:
    def __init__(self, f):
        self.done = False
        self.f = f

    def __call__(self, *args, **kwargs):
        if self.done:
            return None
        else:
            self.done = True
            return self.f(*args, **kwargs)

if __name__ == '__main__':
    class A:
        @lazy_property
        def a(self):
            print('a')
            return 1

        @property
        def b(self):
            print('b')
            return 2

        @run_once
        def initializer(self):
            print('initializer')
            self.c = 3
            self.d = 4

        @initialized_property(initializer)
        def c(self):
            pass

        @initialized_property(initializer)
        def d(self):
            pass

    a = A()
    print(A.a)
    print(A.b)
    print(A.c)
    print(a.a)
    print(a.a)
    print(a.b)
    print(a.b)
    print(a.c)
    print(a.d)
    print(a.c)
    print(a.d)
