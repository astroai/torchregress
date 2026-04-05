import torchregress as tr
import json

def main():
    actual = list(tr.test_time.__all__)
    print(json.dumps(actual, indent=4))

if __name__ == '__main__':
    main()
