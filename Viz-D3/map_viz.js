//Function for processing data and and actions on the map  
function makeMyMaps(collection,lookup){			 
							
  //Transform lag/long coordinates to screen coordinates
  //create path for SVG element & convert GeoJSON to SVG
             
  var transform = d3.geoTransform({point: projectPoint}),
      path = d3.geoPath().projection(transform);   
  
  //Genetate path elementy for each feature and append it to g element    
  var feature = g.selectAll("path") //select all path of element g in SVG
      .data(collection.features)
    .enter()
    .append("path")
    .on("click",function (d) {
      //save current position and fill area
      if (prevElem)  {prevElem.style("fill","");}
      prevElem = d3.select(this);    
      
      d3.select(this).style("fill","red");
    	
    	currLA = lookup.filter(l => {return l.CM_GEOGRAPHY_CODE == d.properties.cmlad11cd   } );
    	//CM_GEOGRAPHY_CODE, GEOGRAPHY_CODE, GEOGRAPHY_NAME
    	
    	//Call update graphs with chosen LA
    	geoCode = currLA[0].GEOGRAPHY_CODE;
    	
    });
    
  
  //Update view when Zoom or pan
  //map.on("viewreset", reset);
  map.on("viewreset", (d)=>{reset(feature,collection,path)});
  
  reset(feature,collection,path);
  
  map.on("click",(d)=>{update(d,path,feature,currLA)});
  
  //update();
};

// Reposition the SVG to cover the features.
function reset(feature,collection,path) {
    //console.log(feature);
    // computing the projected bounding box  
    var bounds = path.bounds(collection),
        topLeft = bounds[0],
        bottomRight = bounds[1];

    svg .attr("width", bottomRight[0] - topLeft[0])
        .attr("height", bottomRight[1] - topLeft[1])
        .style("left", topLeft[0] + "px")
        .style("top", topLeft[1] + "px");

    g.attr("transform", "translate(" + -topLeft[0] + "," + -topLeft[1] + ")");
    
    //initialise the path data by setting the d attribute
    feature.attr("d", path); //pass the path to d attribute (shape, points, ...)
}

// Use Leaflet to implement a D3 geometric transformation.
function projectPoint(x, y) {
    var point = map.latLngToLayerPoint(new L.LatLng(y, x));
    this.stream.point(point.x, point.y);
}
  
var popup = L.popup();
  
function update(d,path,feature,la) {
    //CM_GEOGRAPHY_CODE, GEOGRAPHY_CODE, GEOGRAPHY_NAME
    
    popup.setLatLng(d.latlng)
         .setContent('<p>' + currLA[0].GEOGRAPHY_NAME + ' <br /> ' + d.latlng.toString() + ' </p> ')
         .openOn(map);
    
  	 feature.attr("d",path);
}
  