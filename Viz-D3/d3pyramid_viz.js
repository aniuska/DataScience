/*
Inspired by Jason Davis
https://www.jasondavies.com/d3-pyramid/

*/

/* Setting variables */
var total,
    topMargin = 15,
	 labelSpace = 40,
    innerMargin,
	 outerMargin = 15,
	 commas = d3.format(",.0f"),
	 data_curYear,
	 chartWidth,
	 dataRange;
	 

/*	   
Function:
   pyramid - draw bars to create pyramid graph. Also define combobox for Ethnicity
Parameters:
   data: population counts by age, sex & ethnicity
*/ 
function pyramid (data) {

  //parsing data
  data.forEach(function(d) {
  	                
  						 d.F = +d.F/1000;
  						 d.M = +d.M/1000;
  });
    
  //get data for the current Year 
  data_curYear = data.filter(d => {return d.Year == curYear   } )
 								 .map(obj => {
		  							var row = {};
		  							row['eth'] = obj.Ethnicity;
		  							row['age'] = obj.Age_Range;
		  							row['f'] = obj.F; 
		  							row['m'] = obj.M; 
		  							return row;
							});
							
							
  //Grouping  by Eth to get Ethnicity value 
  var nestData = d3.nest()
						   .key(d => d.Ethnicity)
						   .rollup(v=> v.length)
						   .entries(data);
    
  var selectData = nestData.map(k => k.key);
  selectData.splice(0,1,"All"); //Add option "All"
  
  //Calculate total counts by age range
  totals = calculate_totals(data_curYear);
  
  // Create combobox for Ethnicity
  var filters = d3.select('#filter')
  var span = filters.append('span')
    .text('Select Ethnicity ')
  var yInput = filters.append('select')
      .attr('id','xSelect')
      .on('change',xChange) //call function when values change
    .selectAll('option')
      .data(selectData)
      .enter()
    .append('option')
      .attr('value', function (d) { return d })
      .text(function (d) { return d ;})
  filters.append('p');
  
  /* Setting variables for drawing pyramid graph */  
  var w = 350,
	    h = 200,
	    outerMargin = 15,
	    gap = 2,
	    leftLabel = "Female",
	    rightLabel = "Male";
 
 innerMargin = w/2+labelSpace;
 dataRange = d3.max(totals.map(function(d) { return Math.max(d.m, d.f) }));
  	
 /* Defining bar width & line scale */
 var   barWidth = h / totals.length,
	    yScale = d3.scaleLinear().domain([0, totals.length]).range([0, h-topMargin]);
	        
 chartWidth = w - innerMargin - outerMargin;
 total = d3.scaleLinear().domain([0, dataRange]).range([0, chartWidth - labelSpace]);	
  
 /* main panel */
 var vis = d3.select("#pyramid")
    .append("svg")
    .attr("width", w)
    .attr("height", h);

 /* female label */
 vis.append("text")
  .attr("class", "label")
  .text(leftLabel)
  .attr("x", w-innerMargin)
  .attr("y", topMargin-3)
  .attr("text-anchor", "end");

 /* male label */
 vis.append("text")
  .attr("class", "label")
  .text(rightLabel)
  .attr("x", innerMargin)
  .attr("y", topMargin-3);
  
 /* age Axis label */
 vis.append("text")
  .attr("class", "label")
  .text("Age Range")
  .attr("x", innerMargin - 10)
  .attr("y", topMargin-3)
  .attr("text-anchor", "end");
 

 /* female bars and data labels */ 
 var bar = vis.selectAll("g.bar")
    .data(totals)
  .enter().append("g")
    .attr("class", "bar")
    .attr("transform", function(d, i) {
      return "translate(0," + (yScale(i) + topMargin) + ")";
    });

 var wholebar = bar.append("rect")
    .attr("width", w)
    .attr("height", barWidth-gap)
    .attr("fill", "none")
    .attr("pointer-events", "all");

 var highlight = function(c) {
  return function(d, i) {
    bar.filter(function(d, j) {
      return i === j;
    }).attr("class", c);
  };
 };

 bar
  .on("mouseover", highlight("highlight bar"))
  .on("mouseout", highlight("bar"));

 bar.append("rect")
    .attr("class", "femalebar")
    .attr("height", barWidth-gap);

 bar.append("text")
    .attr("class", "femalebar")
    .attr("dx", -3)
    .attr("dy", "1em")
    .attr("text-anchor", "end");

 bar.append("rect")
    .attr("class", "malebar")
    .attr("height", barWidth-gap)
    .attr("x", innerMargin);

 bar.append("text")
    .attr("class", "malebar")
    .attr("dx", 3)
    .attr("dy", "1em");

 /* sharedLabels */
 bar.append("text")
    .attr("class", "shared")
    .attr("x", w/2)
    .attr("dy", "1em")
    .attr("text-anchor", "middle")
    .text(function(d) { return d.age; });
 
 //Get year value when move slider & update curYear variable   
 d3.select("#year").on("input",function() {updateYear(+this.value,data)}); 

 //Recalculate total counts per chosen year							
 refresh(totals);
}
 
/*
Function: 
   updateScale - Re-calculate data range & scale when data change
   
Parameters: 
   newData - the filtered data
*/    
function updateScale(newData) {
 	dataRange = d3.max(newData.map(function(d) { return Math.max(d.m, d.f) }))
 	total = d3.scaleLinear().domain([0, dataRange]).range([0, chartWidth - labelSpace])
 	
 	console.log(dataRange);
}

/*
Function: 
   xChange - Filter data by the selected option in the combobox. Call updateScale & refresh functions 
   
*/
function xChange() {
    var value = this.value // get the new x value
    console.log(value);
    //filter by Eth
    var filteredData = value == "All"? totals : (
															    data_curYear.filter(d => {return d.eth == value   } )
												    							 .map(obj => {
															  							var row = {};
															  							row['age'] = obj.age;
															  							row['f'] = Math.round(obj.f);
															  							row['m'] = Math.round(obj.m);
															  							return row;
																				})
			  						);
			  						
    //console.log(filteredData);
    updateScale(filteredData);
        
    refresh(filteredData);
    
}
	

//OJO - check this part to find bug in sort labels
/*
Function: 
   refresh - Redraw pyramid bars. Re-calculate bars width & rectangle
   
Parameters: 
   tot - data to show
*/
function refresh(tot) {
	  var bars = d3.selectAll("g.bar")
	      .data(tot);
	  bars.selectAll("rect.malebar")
	    .transition()
	      .attr("width", function(d) { return total(d.m); });
	  bars.selectAll("rect.femalebar")
	    .transition()
	      .attr("x", function(d) { return innerMargin - total(d.f) - 2 * labelSpace; }) 
	      .attr("width", function(d) { return total(d.f); });
	
	  bars.selectAll("text.malebar")
	      .text(function(d) { return commas(d.m); })
	    .transition()
	      .attr("x", function(d) { return innerMargin + total(d.m); });
	  bars.selectAll("text.femalebar")
	      .text(function(d) { return commas(d.f); })
	    .transition()
	      .attr("x", function(d) { return innerMargin - total(d.f) - 2 * labelSpace; });
}

/*
Function: 
   updateYear - Filter original data per a specified year
   
Parameters: 
   year - year value to filter by
*/
function updateYear(year,data) {
    //adjust text on the range slider
    d3.select("#slider_value").text(year);
    d3.select("#slider_value").property("value",year);
    
    //get data for year
    var data_newYear = data.filter(d => {return d.Year == year   } )
 								 .map(obj => {
		  							var row = {};
		  							row['eth'] = obj.Ethnicity;
		  							row['age'] = obj.Age_Range;
		  							row['f'] = obj.F; // ;
		  							row['m'] = obj.M; // /1000;
		  							return row;
							});
    
    var totals = calcalate_totals(data_newYear);
    updateScale(totals);
    console.log(totals);
    
    refresh(totals);
}

/*
Function: 
   calculate_totals - Re-calculate counts 
   
Parameters: 
   mydata - new data to calculate counts
   
Return
  a table with counts by age/age range
*/ 
function calculate_totals(mydata) {
 	
	 	var total_byAge = d3.nest()
						   .key(d => d.age)
						   .sortKeys(d3.descending)
						   .rollup(function(v) { return {
						   	            //"length": v.length,
						   					"f": Math.round(d3.sum(v,function(d) {return d.f})), 
						   					"m": Math.round(d3.sum(v,function(d) {return d.m})),
						   		}})
						   .entries(mydata);	
	  
	   var totals = total_byAge.map(obj => {
	  							var row = {};
	  							row['age'] = obj.key;
	  							row['f'] = obj.value.f;
	  							row['m'] = obj.value.m;
	  							return row;
	  						});
	  						
	  	return totals;
	 
}
  




