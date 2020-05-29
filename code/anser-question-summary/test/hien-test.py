class grandFather():
    print('我是爷爷')


class Parent(grandFather):
    print('我是父类')


class SubClass(Parent):
    print('我是子类')


sub = SubClass()
