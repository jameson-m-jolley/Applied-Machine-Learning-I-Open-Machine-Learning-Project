from regression import get_modle
from data_pipe_extract_features import make_rows
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
def main():
    print("loading the current data and training a model...")
    fn = get_modle()
    print("--------------------------")
    print("welcome to the demo:")
    text = input("input some text:")
    row = make_rows(text)
    pred = fn(row.reshape(1, -1))
    print(f"prediction-> text was AI: {pred[0][0]*100}% human: {pred[0][1]*100}%")
    pass

if __name__ == "__main__":
    main()