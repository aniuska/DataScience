//Asynchronously loading data files: data, GeoJSON data and lookup table
d3.queue()
  .defer(d3.json,"/data/topo/EnglandWales.json")
  .defer(d3.csv,"/data/LAD_lookup.csv")
  .defer(d3.csv,'/data/pop_years_rangeAges.csv')
  .defer(d3.csv,'/data/projections.csv')
  .await(main_viz);

/*********** Global variables ***********/
var curYear = d3.select("#year").property("value");
var currLA,
    geoCode = 'E92000001'; //England

/************** Mapping *************/
//Create base map using OpenStreetView maps centered at London
var map = new L.Map("map", {center: [51.505, -0.1], zoom: 9})
    .addLayer(new L.TileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"));
    
//Allow click on the map and get location
L.svg({clickable:true}).addTo(map);
    
//Synchronising Leaflet & D3 via adding SVG element to overlayPane
//g element ensures  that SVG element & Leaflet layer have the the same common point of reference
var svg = d3.select(map.getPanes().overlayPane).append("svg"),
    g = svg.append("g").attr("class", "leaflet-zoom-hide");

var prevElem;
/************* Pyramid ************************/
var totals;

/************* Projection ************************/

/***************   Functions   ***********************************/

/*
Function: 
   main_viz - Main function. Call all graph function passing data to them
   
Parameters:
	error - Catch error and throw it
	collection - GeoJSON data (boundaries)
	lookup - table with code and names of administrative entities 
	data - simulated population (counts) data
	proj - projection variants data
*/    
//Main function of Spenser visulisation module  
function main_viz(error,collection,lookup,data,proj){
  if (error) throw error;
  
  //Call pyramid main function
  pyramid (data);
  
  //Call map main function
  makeMyMaps(collection,lookup);
  
  //Call projection main function
  projection(proj)
}

function getGeo(geo,lookup) {
	
	currLA = lookup.filter(l => {return l.CM_GEOGRAPHY_CODE == geo  } );

}