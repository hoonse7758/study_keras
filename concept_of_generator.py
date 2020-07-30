# -*- coding: utf-8 -*-


def simple_generator(from_num: int = 0, to_num: int = 10):
    _current = from_num - 1
    while _current < to_num:
        _current += 1
        yield _current


if __name__ == '__main__':

    # 표현식으로 Generator 생성하기
    generator1 = (obj for obj in [1, 2, 3])
    print(f"generator1 = {generator1}")     # 타입 확인 - generator 타입
    for generated in generator1:
        print(f"\tgenerated {generated}")
    try:
        generated = next(generator1)
    except StopIteration:   # 다 꺼내쓰고 비었음
        print(f"next(generator1) = {None}")

    # 함수로 Generator 생성하기
    generator2 = simple_generator(10, 20)
    print(f"generator2 = {generator2}")  # 타입 확인 - generator 타입
    generated: int = 0
    is_generated: bool = True
    while is_generated:
        try:
            generated = next(generator2)
        except StopIteration:
            is_generated = False
        finally:
            print(generated)
