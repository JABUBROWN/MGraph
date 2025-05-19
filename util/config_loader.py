import configparser
import os.path


def load_config(root_path, parser):

    tmp = parser.parse_args()

    data_config = configparser.ConfigParser()
    data_config.read(os.path.join(root_path, "config/data", tmp.data_config_file))  # "./config/data/xxxx.conf"
    model_config = configparser.ConfigParser()
    model_config.read(os.path.join(root_path, "config/model", tmp.model_config_file))  # "./config/model/xxxx.conf"

    parser.add_argument("--data_name", default=data_config["data"]["data_name"], type=str)
    parser.add_argument("--data_file", default=data_config["data"]["data_file"], type=str)
    parser.add_argument("--adj_file", default=data_config["data"]["adj_file"], type=str)
    d = data_config["data"]["adj_file_delimiter"]  # convert '\\0' to a space
    parser.add_argument("--adj_file_delimiter", default=d if d != "\\0" else ' ', type=str)
    parser.add_argument("--id_file", default=data_config["data"]["id_file"], type=str)

    parser.add_argument("--date_range", default=data_config["data"]["date_range"], type=str)
    parser.add_argument("--node_num", default=data_config["data"]["node_num"], type=int)
    parser.add_argument("--daily_time_step", default=data_config["data"]["daily_time_step"], type=int)

    parser.add_argument("--given_time_step", default=int(data_config["data"]["given_time_step"]), type=int)
    parser.add_argument("--predict_time_step", default=data_config["data"]["predict_time_step"], type=int)
    parser.add_argument("--train_ratio", default=data_config["data"]["train_ratio"], type=float)
    parser.add_argument("--val_ratio", default=data_config["data"]["val_ratio"], type=float)

    parser.add_argument("--input_dim", default=data_config["model"]["input_dim"], type=int)
    parser.add_argument("--hidden_dim", default=data_config["model"]["hidden_dim"], type=int)
    parser.add_argument("--embed_dim", default=data_config["model"]["embed_dim"], type=int)
    parser.add_argument("--chebyshev_num", default=data_config["model"]["chebyshev_num"], type=int)

    parser.add_argument("--batch_size", default=model_config["train"]["batch_size"], type=int)
    parser.add_argument("--lr_init", default=model_config["train"]["lr_init"], type=float)
    parser.add_argument("--weight_decay", default=model_config["train"]["weight_decay"], type=float)
    lr_scheduler = model_config["train"]["lr_scheduler"].split(',')
    parser.add_argument("--lr_scheduler", default=model_config["train"]["lr_scheduler"], type=str)
    parser.add_argument("--lr_scheduler_name", default=lr_scheduler[0], type=str)
    parser.add_argument("--gamma", default=lr_scheduler[1], type=float)
    parser.add_argument("--early_stop", default=eval(model_config["train"]["early_stop"]), type=bool)
    parser.add_argument("--early_stop_patience", default=model_config["train"]["early_stop_patience"], type=int)

    parser.add_argument("--scaler", default=model_config["data"]["scaler"], type=str)
    parser.add_argument("--model_name", default=model_config["model"]["model_name"], type=str)
    parser.add_argument("--xavier_init", default=eval(model_config["model"]["xavier_init"]), type=bool)
    parser.add_argument("--optimizer", default=model_config["model"]["optimizer"], type=str)
    parser.add_argument("--loss", default=model_config["model"]["loss"], type=str)
    parser.add_argument("--trainer", default=model_config["train"]["trainer"], type=str)
    parser.add_argument("--clip", default=eval(model_config["train"]["clip"]), type=bool)

    parser.add_argument("--root_path", default=root_path, type=str)
    return parser.parse_args()
