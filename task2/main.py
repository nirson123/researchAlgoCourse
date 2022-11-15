import re
from typing import List as Listt
from doctest import testmod


def email_finder(file: str) -> (Listt[str], Listt[str]):
    """
    finds all emails appearing in a text file. return list of valid emails, and a list of in-valid emails.
    valid email according to the rules in https://help.xmatters.com/ondemand/trial/valid_email_format.htm.
    in-valid email are any word containing a @ symbol but not following the rules.
    :param file: text file to find emails in
    :return: list of valid emails, list of in-valid emails

    >>> email_finder('test.txt')
    (['nir16832@gmail.com', 'gg-gg.j@ffffff.cv'], ['aaa-@mail.com', 'sssff@mail.m'])

    >>> email_finder('test2.txt')
    ([], [])
    """

    # get the text from the file
    with open(file, 'r') as f:
        text = f.read()

    # get all words that contains @
    emails = [word for word in re.split(' |\n|\t', text) if '@' in word]

    # filter valid emails
    valid_email_template = re.compile(
        r'(\d|[A-z]|((\d|[A-z])\.(\d|[A-z]))|((\d|[A-z])_(\d|[A-z]))|((\d|[A-z])-(\d|[A-z])))+@(\d|[A-z]|((\d|[A-z])-(\d|[A-z]))|((\d|[A-z])\.(\d|[A-z])))+\.[A-z]([A-z]+)')
    valid_emails = [email for email in emails
                    if re.match(valid_email_template, email)]

    invalid_emails = sorted(list(set(emails) - set(valid_emails)))

    return valid_emails, invalid_emails


functions_cache = {}    # cache past calls to functions
def lastcall(func):
    """
    decorator that adds cache to a function - if the function was called with same parameter in the past,
    prints a massage. else runs the function.
    :param func: function to run
    :return:
    """

    def inner(x):
        if func.__name__ in functions_cache and repr(x) in functions_cache.get(func.__name__):
            print(f'I already told you that the answer is {functions_cache.get(func.__name__).get(repr(x))}!')
            return
        else:
            func(x)

    return inner


@lastcall
def f(x: int) -> int:
    """
    >>> f(2)
    4
    >>> f(2)
    I already told you that the answer is 4!
    >>> f(10)
    100
    >>> f(2)
    I already told you that the answer is 4!
    >>> f(10)
    I already told you that the answer is 100!
    """
    return x ** 2


class List(list):
    """
    list class that supports multidimensional arrays syntax

    >>> List([[1, 2], [3, 4]])[1, 1]
    4
     >>> List([[1, 2], [3, 4]])[0]
     [1, 2]
    """
    def __init__(self, lst: list):
        super(List, self).__init__(lst)

    # override [] operator
    def __getitem__(self, key):
        if isinstance(key, int):
            return super(List, self).__getitem__(key)
        elif isinstance(key, tuple):
            current = self
            for i in key:
                current = current[i]
            return current
        else:
            raise Exception(f'{type(key)} is not valid')

    # override [] operator for setting values
    def __setitem__(self, key, value):
        if isinstance(key, int):
            super(List, self).__setitem__(key, value)
        elif isinstance(key, tuple):
            current = self
            for i in key[:-1]:
                current = current[i]
            current[key[-1]] = value
        else:
            raise Exception(f'{type(key)} is not valid')


def test_List():
    """
    >>> test_List()
    12
    1
    -9
    [[[1, 2, 3], [4, -9, 6]], [[], [10, 11, 12]]]
    """
    lst1 = List([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ])
    print(lst1[1, 1, 2])
    print(lst1[0, 0, 0])
    lst1[0, 1, 1] = -9
    print(lst1[0, 1, 1])
    lst1[1, 0] = []
    print(lst1)


def main():
    testmod()


if __name__ == '__main__':
    main()
