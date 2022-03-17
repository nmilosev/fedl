# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import flwr
import flwr.common as common

from nus_strategy import NUS
from mnist import MNISTNet, PytorchMNISTClient

from flwr.server.strategy import FedAvg

if __name__ == "__main__":
    model = MNISTNet()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    server_initial_parameters = common.weights_to_parameters(weights)
    # NUS
    flwr.server.start_server(config={"num_rounds": 5}, strategy=NUS(initial_parameters=server_initial_parameters))
    # FedAvg
    # flwr.server.start_server(config={"num_rounds": 5}, strategy=FedAvg(initial_parameters=server_initial_parameters,
    #     fraction_fit = 1,
    #     fraction_eval = 1,
    #     min_fit_clients = 5,
    #     min_eval_clients = 5,
    #    min_available_clients = 5))
