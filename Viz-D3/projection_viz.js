/*
Pojection graph

updated from https://bl.ocks.org/d3noob

data structure
GEOGRAPHY_CODE: "E09000030", 
PROJECTED_YEAR_NAME: "2016", 
OBS_VALUE: "300943", 
VARIANT: "ppp"
*/

function projection(data){
   
   // Parse the date / time
	var parseDate = d3.timeParse("%Y");
	
   // Get the data
	 data.forEach(function(d) {
		d.year = parseDate(d.PROJECTED_YEAR_NAME);
		d.count = +d.OBS_VALUE/1000;
	 });
	//filter by GEOGRAPHY_CODE
	var data_LAD = data.filter(l => {return l.GEOGRAPHY_CODE == geoCode   })
													
   // Set the dimensions of the canvas / graph
	var margin = {top: 30, right: 50, bottom: 70, left: 80},
	    width = 330 - margin.left - margin.right,
	    height = 300 - margin.top - margin.bottom;
	
	// Set the ranges for x & y
	var x = d3.scaleTime().range([0, width]);  
	var y = d3.scaleLinear().range([height, 0]);
	
	// Define the line
	var projline = d3.line()	
	    .x(function(d) { return x(d.year); })
	    .y(function(d) { return y(d.count); });
	    
	// Adds the svg canvas
	var svg = d3.select("#projection")
	    .append("svg")
	        .attr("width", width + margin.left + margin.right)
	        .attr("height", height + margin.top + margin.bottom)
	        .attr("id", "svg_proj")
	    .append("g")
	        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	
	 // Scale the range of the data
	 x.domain(d3.extent(data_LAD, function(d) { return d.year; }));
	 //y.domain([0, d3.max(data_LAD, function(d) { return d.count; })]);
	 y.domain([d3.min(data_LAD, function(d) { return d.count; }), d3.max(data_LAD, function(d) { return d.count; })]);
	
	 // Nest the entries by VARIANT
	 var dataNest = d3.nest()
	     .key(function(d) {return d.VARIANT;})
	     .entries(data_LAD);
	 
	 // set the lines colour array
	 var color = d3.scaleOrdinal(d3.schemeCategory10);
	
	 legendSpace = width/dataNest.length; // spacing for the legend
	
	 // Loop through each VARIANT / key
	 dataNest.forEach(function(d,i) { 
	     svg.append("path")
	         .attr("class", "line")
	         .style("stroke", function() { // Add the colours dynamically
	             return d.color = color(d.key); })
	         .attr("d", projline(d.values));
	
	     // Add the Legend
	     svg.append("text")
	         //.attr("x", (legendSpace/2)+i*legendSpace)  // space legend
	         //.attr("y", height + (margin.bottom/2)+ 2)
	         .attr("x", width )  // space legend
	         .attr("y", height/2 + 20*i +20)
	         .attr("class", "legend_proj")    // style the legend
	         .style("fill", function() { // Add the colours dynamically
	             return d.color = color(d.key); })
	         .text(d.key); 
	
	 });
	
	  // Add the X Axis
	  svg.append("g")
	   .attr("class", "axis_proj")
	   .attr("transform", "translate(0," + height + ")")
	   .call(d3.axisBottom(x));
	
	  // Add the Y Axis
	  svg.append("g")
	   .attr("class", "axis_proj")
	   .call(d3.axisLeft(y));
	   
	  //Add axis label
	  svg.append("text")
	         .attr("transform", "translate(" + (width/2) + "," + (height + margin.bottom/2 ) + ")")  // space legend
	         .attr("text-anchor", "middle")
	         .text("Years"); 
	         
	  svg.append("text")
	         .attr("transform", "rotate(-90)")  
	         .attr("x", 0 - (height/2))
	         .attr("y", 0 - margin.left + 15 )
	         .attr("dy", "1em")    
	         .style("text-anchor","middle" )
	         .text("Persons"); 
	  


   
   
   
}