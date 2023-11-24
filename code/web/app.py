import json

import pandas as pd
from flask import Flask, render_template, request

from helpers.data import categories_dict, countries_dict, fuel_types_dict

app = Flask("co2e")


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        country = request.form.get("country")
        make = request.form.get("make")
        cn = request.form.get("cn")
        fuel_type = request.form.get("ft")
        year = request.form.get("year")
    else:
        country = request.args.get("country")
        make = request.args.get("make")
        cn = request.args.get("cn")
        fuel_type = request.args.get("ft")
        year = request.args.get("year")

    return render_template(
        "index.html",
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
