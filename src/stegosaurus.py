'''
Main module to encode and decode messages using the Stegosaurus algorithm.
'''

import argparse


def encode(message: str) -> str:
    '''Takes message to be encoded as string, returns cover text as string.'''

    return message


def decode(cover_text: str) -> str:
    '''Takes cover text message as string, extracts and returns secret as string.'''

    return cover_text


if __name__ == '__main__':

    # Take message or cover text as command line arguments
    parser = argparse.ArgumentParser(description='Command line stegosaurus entry-point')
    parser.add_argument('-e', '--encode', type=str, help='Message to be encoded')
    parser.add_argument('-d', '--decode', type=str, help='Cover text to be decoded')
    args = parser.parse_args()

    if args.encode:
        print(encode(args.encode))

    if args.decode:
        print(decode(args.decode))
