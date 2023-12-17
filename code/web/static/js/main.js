// Création de la variable contenant le graphique des émissions, avec une portée globale
let forecast_chart;

/**
 * Crée ou met à jour le graphique des émissions
 * @param {float} co2e - Les émissions de GES pour la partie thermique du véhicule
 * @param {float} z - La consommation électrique du véhicule
 */
function updateChart(co2e, z){
    // Initialise les années de début et de fin de la période à afficher
    const year_from = document.getElementById("year").value;
    const year_to = new Date().getFullYear() + 15;
    // Récupère les dimensions du graphique
    const chart_height = document.getElementById("chart-canvas").height;
    const chart_width = document.getElementById("chart-canvas").width;

    // Affiche une animation de chargement
    document.getElementById("chart").innerHTML = '<div class="spinner-grow text-primary" role="status" height="' + chart_height + '" width="' + chart_width + '"><div class="sr-only"></div></div>'

    // Récupère les données d'émissions auprès de l'API
    fetch("http://" + (server_ip_address != client_ip_address ? server_ip_address : "localhost") + ":8000/forecast?year_from=" + year_from + "&year_to=" + year_to,{
            headers : {'Content-Type': 'application/json', 'Accept': 'application/json'}
        })
        .then((response) => {
            if (!response.ok) {throw new Error('forecast@api a retourné une réponse non exploitable');}
            return response.json();;
        })
        .then((json) => {
            let chart_labels = [];
            let chart_values = [];
            // Création des données du graphique
            // co2e en gCO2e/km, z en Wh/km et json[i] en gCO2e/KWh
            for (let i = year_from; i <= year_to; i++) {
                chart_labels.push(i);
                chart_values.push(co2e + (json[i] * z / 1000));
            }
            // Crée le graphique
            document.getElementById("chart").innerHTML = '<canvas class="my-4 w-100" id="chart-canvas" width="800" height="300"></canvas>';
            if(chart_values.length>1){
                if(forecast_chart){forecast_chart.destroy();}
                const chart_canvas = document.getElementById('chart-canvas');
                forecast_chart = new Chart(chart_canvas, {
                    type: 'line',
                    data: {
                        labels: chart_labels,
                        datasets: [{
                            data: chart_values,
                            lineTension: 0,
                            backgroundColor: 'transparent',
                            borderColor: '#007bff',
                            borderWidth: 4,
                            pointBackgroundColor: '#007bffaa'
                        }]
                    },
                    options: {
                        plugins: {
                            legend: {
                                display: false
                            },
                            title: {
                                display: true,
                                font: {
                                    size: 18
                                },
                                text: "Prédictions des émissions de GES (gCO2e/km) pour ce véhicule"
                            },
                            tooltip: {
                                boxPadding: 3
                            }
                        },
                    }
                })
            }
        })
        .catch((error) => {console.error('updateChart a rencontré une erreur pour récupérer les données: ', error);});
}

/**
 * Ajoute les différentes options dans une liste déroulante
 * @param {Array} arr - Liste des options à ajouter
 * @param {string} selectType - Type du select (identifiant de celui-ci)
 * @param {Object} dict - (Optionnel) Dictionnaire permettant d'associer un texte à une valeur
 */
function populateSelect(arr, selectType, dict=null){
    for (let i = 0; i < arr.length; i++) {
        let elt = arr[i].toString();
        if(elt.trim().length>1) {
            let opt = document.createElement("option");
            opt.textContent = dict?dict[elt]:elt;
            opt.value = elt;
            document.getElementById(selectType).appendChild(opt);
        }
    }
}

/**
 * Met à jour les listes de recherche ou les informations sur le véhicule
 * @param {string} caller - l'identifiant du contrôle qui a fait appel à la fonction
 */
function updateData(caller){
    // Récupère les informations en appelant l'API
    fetch("http://" + (server_ip_address != client_ip_address ? server_ip_address : "localhost") + ":8000/data?caller=" + caller
        + "&country=" + document.getElementById("country").value
        + "&make=" + document.getElementById("make").value
        + "&cn=" + document.getElementById("cn").value
        + "&ft=" + document.getElementById("ft").value
        + "&year=" + document.getElementById("year").value,{
            headers : {'Content-Type': 'application/json', 'Accept': 'application/json'}
        })
        .then((response) => {
            if (!response.ok) {throw new Error('data@api a retourné une réponse non exploitable');}
            return response.json();
        })
        .then((json) => {
            switch(caller){
                case "country":
                    document.getElementById("make").options.length = 0;
                    document.getElementById("cn").options.length = 0;
                    document.getElementById("ft").options.length = 0;
                    document.getElementById("year").options.length = 0;
                    populateSelect(json.makes, "make");
                    break;
                case "make":
                    document.getElementById("cn").options.length = 0;
                    document.getElementById("ft").options.length = 0;
                    document.getElementById("year").options.length = 0;
                    populateSelect(json.cns, "cn");
                    break;
                case "cn":
                    document.getElementById("ft").options.length = 0;
                    document.getElementById("year").options.length = 0;
                    populateSelect(json.fts, "ft", fuel_types_dict);
                    break;
                case "ft":
                    document.getElementById("year").options.length = 0;
                    populateSelect(json.years, "year");
                    break;
                case "year":
                    const vehicle = json.vehicle[0];
                    document.getElementById("info-make").innerHTML = " " + document.getElementById("make").value;
                    document.getElementById("info-cn").innerHTML = " " + document.getElementById("cn").value;
                    document.getElementById("info-year").innerHTML = " " + document.getElementById("year").value;
                    document.getElementById("info-country").innerHTML = " " + countries_dict[document.getElementById("country").value];
                    document.getElementById("info-category").innerHTML = vehicle[0]?" " + vehicle[0] + " (" + categories_dict[vehicle[0]] + ")":" N/C";
                    document.getElementById("info-mro").innerHTML = vehicle[1]?" " + vehicle[1] + " kg":" N/C";
                    document.getElementById("info-length").innerHTML = vehicle[3]?" " + vehicle[3] + "mm":" N/C";
                    document.getElementById("info-width").innerHTML = vehicle[4]?" " + vehicle[4] + " mm":" N/C";
                    document.getElementById("info-ft").innerHTML = " " + fuel_types_dict[document.getElementById("ft").value];
                    document.getElementById("info-ec").innerHTML = vehicle[5]?" " + vehicle[5] + " cm<sup>3</sup>":" N/C";
                    document.getElementById("info-co2e").innerHTML = vehicle[2]?" " + vehicle[2] + " g/km":" N/C";
                    document.getElementById("info-z").innerHTML = vehicle[6]?" " + vehicle[6] + " Wh/km":" N/C";
                    let ecoscore = "none";
                    let ges = vehicle[2] + (263 * vehicle[6] / 1000);
                    if(ges < 50)
                        ecoscore = "a";
                    else if(ges < 100)
                        ecoscore = "b";
                    else if(ges < 150)
                        ecoscore = "c";
                    else if(ges < 200)
                        ecoscore = "d";
                    else if(ges >= 200)
                        ecoscore = "e";
                    document.getElementById("ecoscore").src = "static/img/ecoscore-" + ecoscore + ".png";
                    updateThumbnail();
                    updateChart(vehicle[2], vehicle[6]);
            }
            // S'il n'y a qu'un élément, simule le clic sur celui-ci pour mettre à jour les listes suivantes
            if(json[Object.keys(json)[0]].length == 1 && Object.keys(json)[0].substring(0, Object.keys(json)[0].length-1)!=caller){
                updateData(Object.keys(json)[0].substring(0, Object.keys(json)[0].length-1));
            }
        })
        .catch((error) => {console.error('updateData a rencontré une erreur pour récupérer les données: ', error);});
}

/**
 * Met à jour l'image miniature illustrant le véhicule'
 */
function updateThumbnail(){
    const make = document.getElementById("make").value;
    const cn = document.getElementById("cn").value;
    const year = document.getElementById("year").value;

    // Affiche une animation de chargement
    document.getElementById("vehicle-thumbnail").innerHTML = '<div class="spinner-grow text-primary" role="status"><span class="sr-only"></span></div>'

    // Récupère l'URL de la miniature auprès de l'API
    fetch("http://" + (server_ip_address != client_ip_address ? server_ip_address : "localhost") + ":8000/data/thumbnail?vehicle=" + make + " " + cn + " " + year,{
            headers : {'Content-Type': 'application/json', 'Accept': 'application/json'}
        })
        .then((response) => {
            if (!response.ok) {throw new Error('thumbnail@api a retourné une réponse non exploitable');}
            return response.json();
        })
        .then((json) => {
            if(json.thumbnail_url.length > 0){
                document.getElementById("vehicle-thumbnail").innerHTML = '<img id="vehicle-thumbnail-img" class="img-thumbnail rounded" src="' + json.thumbnail_url + '">';
            }
            else{console.log(json.error);}
        })
        .catch((error) => {console.error('updateThumbnail a rencontré une erreur pour récupérer les données: ', error);});
}
