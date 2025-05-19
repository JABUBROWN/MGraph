import os.path
import os

class Logger:

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fn = "{}_{}_seed{}.txt".format(args.model_name, args.data_name, args.seed)
        self.path = os.path.join(self.args.root_path, "log", self.fn)
        # Create log directory if it doesn't exist
        log_dir = os.path.join(self.args.root_path, "log")
        os.makedirs(log_dir, exist_ok=True)

    def log(self, text):
        with open(self.path, "a") as f:
            f.write(text + '\n')

    def log_and_print(self, text):
        self.log(text)
        print(text)
