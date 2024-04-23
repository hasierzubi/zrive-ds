import time
from urllib.parse import urlencode
import requests
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

from typing import Dict


logger =logging.getLogger(__name__)

logger.level=logging.INFO

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790}
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

def get_data_meteo_api(longitude: float, latitude: float, start_date: str, end_date: str):
    headers= {}

    params= {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "models": "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S",#NICAM16_85
        "daily": VARIABLES,
    }
    #url = API_URL + urlencode(params, safe=",")
    #print("URL:", url) 
    return request_with_cooloff(API_URL + urlencode(params, safe=","),headers)



def _request_with_cooloff(
        url: str, headers: Dict[str, any], num_attempts: int,  payload: Dict[str, any] = None
    ):       
       
       
    cooloff=1
    for call_count in range(num_attempts):
        try:
            if payload is None:
                response= requests.get(url,headers=headers)
                #print("ha llegado")
                #print(response)
            else:
                response=requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
 
        except requests.exceptions.ConnectionError as e:
            logger.info("API refused coonnection")
            logger.warning(e)
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff*=2
                continue
            else: 
                raise
        

        except requests.exceptions.HTTPError as e:
            logger.warning(e)
            if response.status_code == 404:
                raise

            if response.status_code == 400:
                print("algo ha fallado, la url es:")
                print(url)
                raise

            logger.info(f"API return code {response.status_code} cooloff at {cooloff}") #la f para hacer el sring
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff*=2
                continue
            else:
                raise
 
        return response
    

def request_with_cooloff(
    url: str,
    headers: Dict[str, any],
    payload: Dict[str, any] = None,
    num_attempts: int = 10,
) -> Dict[Any, Any]:
    return json.loads(
        _request_with_cooloff(
            url,
            headers,
            num_attempts,
            payload,
        ).content.decode("utf-8")
    )

def compute_variable_mean_and_std(data: pd.DataFrame):
    calculate_ts = data[["city", "time"]].copy()
    variable = "temperature_2m_mean"
    idxs = [col for col in data.columns if col.startswith(variable)]
    calculate_ts[f"{variable}_mean"] = data[idxs].mean(axis=1)
    calculate_ts[f"{variable}_std"] = data[idxs].std(axis=1)
    return calculate_ts
 



def plot_timeseries(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data["year"] = pd.to_datetime(data["time"]).dt.year

    variable = 'temperature_2m_mean'
    for k, city in enumerate(data.city.unique()):
        city_data = data[data['city'] == city].copy()
        city_data['mid_'] = city_data[f"{variable}_mean"]
        city_data['upper_'] = city_data[f"{variable}_mean"] + city_data[f"{variable}_std"]
        city_data['lower_'] = city_data[f"{variable}_mean"] - city_data[f"{variable}_std"]
        
        city_data.groupby("year")["mid_"].apply("mean").plot(
            ax=ax, label=f"{city}", color=f"C{k}"
        )
        city_data.groupby("year")["upper_"].apply("mean").plot(
            ax=ax, ls="--", label="_nolegend_", color=f"C{k}"
        )
        city_data.groupby("year")["lower_"].apply("mean").plot(
            ax=ax, ls="--", label="_nolegend_", color=f"C{k}"
        )
    
    ax.set_title(f"Temperature Time Series for {variable}")
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    
    plt.tight_layout()
    plt.savefig("/home/hasierza/zrive-ds/src/module_1/clima.png", bbox_inches="tight")
    plt.show()


def main():
    city = "Madrid"
    coordinates = COORDINATES[city]
    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]
    
    data_list = []
    start_date = "2010-01-01"
    end_date = "2014-12-05"
    time_span = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()

    for k in range(len(time_span) - 1):
        partial_start = time_span[k]
        partial_end = time_span[k + 1]
        data_list.append(
            (pd.DataFrame(get_data_meteo_api(longitude, latitude, partial_start, partial_end)["daily"]).assign(city=city))
        )
        print("estoy en", k)

    data = pd.concat(data_list)
    print(data.head())

    # Calcula la media y la desviación estándar solo para la variable temperature_2m_mean
    calculated_ts = compute_variable_mean_and_std(data)[["city", "time", "temperature_2m_mean_mean", "temperature_2m_mean_std"]]

    print(calculated_ts.head())

    plot_timeseries(calculated_ts)


if __name__ == "__main__":
    main()