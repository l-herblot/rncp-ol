import json

import pandas as pd
from flask import Flask, render_template, request

from helpers.data import categories_dict, countries_dict, fuel_types_dict
from helpers.network import get_public_ip

app = Flask("co2e")


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        country = request.form.get("country")
    else:
        country = request.args.get("country")

    return render_template(
        "index.html",
        server_ip_address=get_public_ip(),
        countries=countries_dict,
        countries_dict_str="{"
        + ",".join(
            [
                f"'{country_code}':'{country}'"
                for country_code, country in countries_dict.items()
            ]
        )
        + "}",
        country=country or list(countries_dict.values())[0],
        fuel_types_dict_str="{"
        + ",".join([f"'{ft}':'{ft_fr}'" for ft, ft_fr in fuel_types_dict.items()])
        + "}",
        categories_dict_str="{"
        + ",".join([f"'{ct}':'{ct_desc}'" for ct, ct_desc in categories_dict.items()])
        + "}",
    )


######
###
#
# json.dumps()
#
###
#####

if __name__ == "__main__":
    app.run(host="0.0.0.0")
