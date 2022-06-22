import flwr
import flwr.common as common
import os
import sys
import time
import traceback


from flwr.server.strategy import FedAvg

if __name__ == "__main__":
    print(os.environ)

    try:
        if "--fedavg" in sys.argv:
            # FedAvg
            flwr.server.start_server(
                config={"num_rounds": 5},
                strategy=FedAvg(
                    fraction_fit=1,
                    fraction_eval=1,
                    min_fit_clients=2,
                    min_eval_clients=2,
                    min_available_clients=2,
                ),
            )
        else:
            from nus_strategy import NUS
            # NUS
            flwr.server.start_server(
                config={"num_rounds": 5},
                strategy=NUS(
                    fraction_fit=1,
                    fraction_eval=1,
                    min_fit_clients=2,
                    min_eval_clients=2,
                    min_available_clients=2,
                ),
            )
    
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    if "GRACEFUL_EXIT" in os.environ and os.environ["GRACEFUL_EXIT"] == "disable":
        print("FL client finished, not terminating the process due to GRACEFUL_EXIT")
        time.sleep(3600)
        
