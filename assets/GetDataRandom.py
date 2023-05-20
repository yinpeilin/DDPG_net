import json
import random
import time


def GetDeviceLocationRandom(json_data):

    # set the seed so that the results of the random number generator are the same each time for this function call
    random.seed(time.time())
    for i in range(json_data["device"]["num_max"]):

        x_data = random.randint(
            json_data["device"]["location_x_limit"][0], json_data["device"]["location_x_limit"][1])
        y_data = random.randint(
            json_data["device"]["location_y_limit"][0], json_data["device"]["location_y_limit"][1])

        if len(json_data["device"]["amount_of_information"]) > i:
            json_data["device"]["location"][i][0] = x_data
            json_data["device"]["location"][i][1] = y_data
        else:
            json_data["device"]["location"].append([x_data, y_data])

    return json_data


def GetDeviceInformationAmountRandom(json_data):

    # set the seed so that the results of the random number generator are the same each time for this function call
    random.seed(time.time())
    for i in range(json_data["device"]["num_max"]):
        amount_information = random.random()*(json_data["device"]["amount_of_information_limit"][1] - json_data["device"]
                                              ["amount_of_information_limit"][0]) + json_data["device"]["amount_of_information_limit"][0]

        if len(json_data["device"]["amount_of_information"]) > i:
            json_data["device"]["amount_of_information"][i] = amount_information
        else:
            json_data["device"]["amount_of_information"].append(
                amount_information)

    return json_data


def ReadData(data_path):
    json_data = json.load(open(data_path, "r"))

    return json_data


def WriteBack(json_data, data_path):

    json.dump(json_data, open(data_path, "w"))

    pass


if __name__ == "__main__":

    data_path = "assets/data.json"
    data_save_path = "assets/data_random.json"
    json_data = ReadData(data_path)
    json_data = GetDeviceLocationRandom(json_data)
    json_data = GetDeviceInformationAmountRandom(json_data)
    WriteBack(json_data, data_save_path)
