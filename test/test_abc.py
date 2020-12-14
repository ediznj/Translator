from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def test(self):
        pass
    #
    # @abstractmethod
    # def get_info(self, name: str) -> str:
    #     print('calling abstract method')


class Derived(Base):
    def test(self):
        print('Abstract method impl')


if __name__ == '__main__':
    # d = Derived()
    b = Base()
    # d.test()
    b.get_info('test')
