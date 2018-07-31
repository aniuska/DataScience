# GWR for Lars Results
#Variables name and index are on lar_vars.csv

library("maptools")
library("rgdal")
library("rgeos")
library("GWmodel") 
#library("gwrr")
library("dplyr")
library("RColorBrewer")


#Setting working directory
setwd("C:\\Users\\Ana\\Dropbox\\__City University\\__Disertation\\code-scripts\\")

shapefiles = c("england_lad","London_Boroughs","London_Ward","London_LSOA")
unempfiles = c("Census_Per_LA.csv","Census_Per_London.csv","Census_Per_Ward_London.csv","Census_Per_LSOA_London.csv")

#shapefiles =c("London_Ward")
#unempfiles =c("Census_Per_Ward_London.csv")

data_path= 'data'
shapefiles_path = 'shapefiles'

pop_ix = 3
den_ix = 9
y_ix = 48

#get the variables' name form .csv file
var_names <- read.csv("larLasso_vars.csv",stringsAsFactors = FALSE)

mypalette.6 <- brewer.pal(6, "Spectral")

#name = "Census_Per_LA.csv"
#loop
for ( name in unempfiles) {
  #Read non-spatial data from .csv
  filename  = paste(data_path,name,sep="/")
  unemp_data <- read.csv(filename,
                         stringsAsFactors = FALSE,
                         header = TRUE,
                         check.names = FALSE
                         )
  #Get variable names to be used (header of columns)
  census = strsplit(name, "[.]")[[1]]
  #var_names$varName[var_names$Filename==census[1]]
  s<-c(colnames(unemp_data)[0:pop_ix],colnames(unemp_data)[y_ix],var_names$varName[var_names$Filename==census[1]])
  cols = unlist(strsplit(paste(s,collapse=','),",")[[1]])
  
  #remove unnecessary columns
  data = subset(unemp_data, select = cols)
  
  data[5:length(cols)]<-scale(data[5:length(cols)])
  data$k045<-scale(data$k045,scale=FALSE)

  #Read shapefile as spatial Dataframe
  layer = shapefiles[which(unempfiles == name)]
  geo <- readOGR(dsn="shapefiles",layer=layer)#same as readShapePoly
  #geo@data

  ################ GWR #############################
  #Join dataframe unemp_data to spatial Dataframe
  geo@data <- left_join(geo@data, data, by = c('CODE'))
 
  #Distance matrix, kernel and bandwidth
  #DM <- gw.dist(dp.locat = coordinates(map))
  #DMC <- gw.dist(dp.locat = centroids)
  
  #Createformula
  f<-unlist(strsplit(paste(var_names$varName[var_names$Filename==census[1]],collapse=','),",")[[1]])
  frm<-as.formula(paste('k045', paste(f, collapse=" + "), sep=" ~ "))

  #Bandwidth selection for basic GWR
  bw.gwr.1 <- bw.gwr(frm, data = geo, approach = "CV",
      	kernel = "gaussian", adaptive = TRUE)
  bw.gwr.1

  
  gwr.res <- gwr.basic(frm, data = geo, 
                       bw = bw.gwr.1, 
      	           kernel = "gaussian", 
                       adaptive = TRUE )#, F123.test = TRUE
  #gwr.res
  file_name = paste("GWRresults_lars",census[1],sep="_")
  file_name = paste("lars",file_name,sep="/")
  writeGWR(gwr.res,fn=file_name) 
  #cat(cat("========= GWR Results for ",census[1],"==================="),
       #gwr.res$SDF@data,file="output_GWR_Lars.txt",sep="\n",append=TRUE)
  
  plTitle = paste("Lars",layer,sep="-")

  #Mapping values for each variable
  file_name = paste("GWR_lars",census[1],"Vars",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  #breakpoints <- c(0,10,20,30,40,50,60)
  print(spplot(geo,f,col = "transparent",at = breakpoints,
               col.regions = brewer.pal(8,"Reds"),cut=6,
               par.settings = list(panel.background=list(col="gray")),
               main=paste(plTitle,"variables value")
              )
       )
  dev.off()
  
  #Mapping coeffcient estimates for each variable
  file_name = paste("GWR_lars",census[1],"Coefs",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  #x11()
  print(spplot(gwr.res$SDF, f, key.space = "right", 
	         col.regions = brewer.pal(8,"Blues"),cuts=5,
               par.settings = list(panel.background=list(col="gray")),
	         main = paste("GWR coefficient estimates",plTitle)
              )#col.regions = mypalette.6, at = c()
       )
  dev.off()

  #x11()
  #Mapping studentised residual for rersponse
  file_name = paste("GWR_lars",census[1],"Stud","Residuals",sep="_")
  file_name = paste(file_name,"png",sep=".")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  file_name = paste("lars",file_name,sep="/")
  print(spplot(gwr.res$SDF,"Stud_residual", key.space = "right", 
	      col = "transparent",cuts = 5,
            col.regions=brewer.pal(8,"Reds"),
            par.settings = list(panel.background=list(col="gray")),
	      main = paste("GWR studentised residuals",plTitle)
        )
       )#col.regions = mypalette.6
  dev.off()

  #x11()
  p = paste(f,'_TV',sep="")
  #Mapping pseudo t-statistic and p-value
  #Mapping p-values for each variable
  file_name = paste("GWR_lars",census[1],"t-values",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  print(spplot(gwr.res$SDF,p, key.space = "right", 
	          col = "transparent",cuts=5,
                col.regions=brewer.pal(8,"Blues"),
                par.settings = list(panel.background=list(col="gray")),
	          main = paste("GWR t-values",plTitle)
          )#col.regions = mypalette.6,
       )
  dev.off()

  #x11()
  #p-value significant test: p-values adjusted of GWR outputs  
  pvalue<-gwr.t.adjust(gwr.res) 
  pvalueTable<-pvalue$SDF@data 
  #names(pvalueTable)

  p = paste(f,'_p',sep="")
  mypalette.gwr.mht <- brewer.pal(4, "Spectral")
  #X11(width=10,height=12)
  file_name = paste("GWR_lars",census[1],"O-p-values",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  print(spplot(pvalue$SDF, p, key.space = "right", at = c(0, 0.025, 0.05, 0.1, 1.00), 
         col = "transparent", col.regions=brewer.pal(8,"Reds"), cuts=6,
         par.settings = list(panel.background=list(col="gray")),
         main = paste("GWR Original p-values",plTitle)
       )
     )
  #col.regions = mypalette.gwr.mht,
  dev.off()

  #X11(width=10,height=12)
  p = paste(f,'_p_bh',sep="")
  file_name = paste("GWR_lars",census[1],"Benjamini-Hochberg",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  print(spplot(pvalue$SDF, p, key.space = "right", col = "transparent", 
              at = c(0, 0.025, 0.05, 0.1, 1.0000001), 
              col.regions=brewer.pal(8,"Reds"),cuts=6,
              par.settings = list(panel.background=list(col="gray")),
              main = paste("p-values adjusted by Benjamini-Hochberg",plTitle)
         )
   )
  dev.off()
 
  #local collinearity test
  gwr.coll.data <- gwr.collin.diagno(frm,data = geo, bw = bw.gwr.1, 
                                      kernel = "gaussian", adaptive = TRUE)
  
  mypalette.coll.2 <-brewer.pal(6,"PuBuGn")
  #X11(width=10,height=12)
  p = paste(f,'_VIF',sep="")
  file_name = paste("GWR_lars",census[1],"VIF",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  print(spplot(gwr.coll.data$SDF,p,key.space = "right", 
         col.regions = brewer.pal(8,"Greens"),
         par.settings = list(panel.background=list(col="gray")),
         at=c(7,8,9,10,11,12,13),col="transparent",
              main=paste("Local VIFs",plTitle)
        )#,col="transparent", colorkey = T
       )
  dev.off()

  mypalette.coll.3 <-brewer.pal(8,"Reds")
  #X11(width=10,height=12)
  #p = paste(f,'_VIF',sep="")
  file_name = paste("GWR_lars",census[1],"CN",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("lars",file_name,sep="/")
  png(filename=file_name, width = 900, height = 600, units = 'px')
  print(spplot(gwr.coll.data$SDF,"local_CN",key.space = "right",
          col.regions=brewer.pal(8,"Reds"),cuts=5,
          par.settings=list(fontsize=list(text=15),panel.background=list(col="gray")),
          main=list(label=paste("Local condition numbers",plTitle), cex=1.25)

              )
       )
  dev.off()

}




