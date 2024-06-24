function updateOpacity(value) {
    var opacity = value / 100;
    var overlays = document.getElementsByClassName('leaflet-image-layer');
    for (var i = 0; i < overlays.length; i++) {
        overlays[i].style.opacity = opacity;
    }
}

class overlayHandler{
    overlays = {};
    map = null;
    currentlyChecked= [];
    currentType = null;
    currentSat = null;
    
    constructor(map){
        this.map = map;
        this.setListener();
    }
    
    
    addOverlays(overlays){
        this.overlays = overlays;
    }
    
    addOverlay(first_year,last_year,country,clusters,bounds){
        this.overlays[country] = new overlay(first_year,last_year,country,clusters,bounds[0],bounds[1]);
    }
    
    updateMap(year){
        if(this.currentSat && this.currentType){
            for (const country of this.currentlyChecked){
                this.overlays[country].updateMap(this.map,this.currentType,this.currentSat,year)
            }
        }
    }
    
    removeFromMap(){
        for (const country of this.currentlyChecked){
            this.overlays[country].removeCurrentOverlay(this.map,this.currentType,this.currentSat,year)
        }
    }
    
    setListener() {
        this.setMapListener()
        this.setSliderListener()
        this.setCheckboxesListener()
        this.setRadioListener(document.getElementsByName("sat"),this.updateSat.bind(this))
        this.setRadioListener(document.getElementsByName("typeVis"),this.updateTypeVis.bind(this))
    }

    setMapListener(){
        this.map.on('click', (e)=>{
            L.popup()
            .setLatLng(e.latlng)
            .setContent(e.latlng.toString())
            .openOn(this.map);
        });
    }
    
    setSliderListener(){
        let handler = this
        $('#year').on('input', function() {
            if(handler.getType()!="kmeans")handler.updateMap($(this).val());
            let span = document.getElementById("yearValue");
            span.innerHTML = "";
            span.appendChild(document.createTextNode($(this).val()));
        });
    }
    
    setCheckboxesListener(){
        let handler = this
        var checkboxes = [...document.getElementsByClassName('leaflet-control-layers-selector')]
                                                    .filter(check => check.type == "checkbox");
        for (var i = 0; i < checkboxes.length; i++){
            checkboxes[i].addEventListener('change', function() {
                let innerText = this.nextSibling.textContent.trim();
                if (this.checked) {
                    handler.notifyCheck(innerText);
                    for (let color of ["red","purple"])
                        handler.addMarkerListener(color);
                }else{
                    handler.notifyUncheck(innerText);
                }
            });   
        }
    }
    
    setRadioListener(radioList, callback){
        for(const radio of radioList){
            radio.addEventListener('change',()=>{
                if (radio.checked){
                    callback(radio.value === "null" ? null :radio.value)
                }
            });
        }  
    }
    
    
    update(){
        this.updateMap($('#year').val())
    }
    
    addMarkerListener(color){
        let nameMarker = "awesome-marker-icon-"+color+" awesome-marker leaflet-zoom-animated leaflet-interactive"
        const markers = [...document.getElementsByClassName(nameMarker)].filter(m => !m.dataset.hasEvent)
        for (let marker of markers){
            marker.addEventListener("click",()=>{
                let cpt = 0;
                let idInterval = setInterval(()=>{
                    cpt++;
                    console.log("Tour : ", cpt)
                    let img = document.getElementById(getImgName(color));
                    if(img){
                        if (color == "purple")updateHist($('#year').val());
                        if (this.currentSat)updateImage(this.currentSat);
                        clearInterval(idInterval);
                    }
                    if(cpt>=100)clearInterval();
                },10)
            });
            marker.dataset.hasEvent = true
        }
    }
    
    notifyCheck(country){
        this.currentlyChecked.push(country);
        this.update();
        
    }
    
    notifyUncheck(country){
        let del = this.currentlyChecked.splice(this.currentlyChecked.indexOf(country),1);
        if (this.overlays[del])
            this.overlays[del].removeCurrentOverlay(this.map);
    }
    
    
    updateTypeVis(typeVis){
        this.currentType = typeVis;
        typeVis == null ? this.removeFromMap() : this.update();
    }
    
    getType(){
        return this.currentType
    }
    
    updateSat(sat){
        console.log(this,sat)
        this.currentSat = sat;
        console.log("Nouveau sat",this.currentSat , sat)
        sat==null?this.removeFromMap():this.update();
    }
    
    
}

class overlay{
    imageOverlays = {"ntl_intensity":{},"kmeans":{}};
    currentOverlay = null;
    
    constructor(first_year, last_year, country, clusters, top_map, bot_map) {
        let path ="";
        const sats = ["DMSP", "VIIRS"];
        for (let sat of sats) {
            this.imageOverlays["ntl_intensity"][sat] = {}
            path = "../analysis/" + country;
            for (let i = first_year; i < last_year+1; i++) {
                let n = "/ntl_intensity_" + i + "_" + sat + ".png";
                this.imageOverlays["ntl_intensity"][sat][i] = L.imageOverlay(path + '/ntl_intensity/' + sat + "/" + n,
                    [[bot_map[0], top_map[1]], [top_map[0], bot_map[1]]],
                    {opacity: 0.6}
                );  
            }
            this.imageOverlays["kmeans"][sat] = L.imageOverlay(
                path  + "/kmeans_analysis/"+ sat + "/" + clusters + "_" + first_year + "-" + last_year + "/cluster_img_"+clusters +".png",
                [[bot_map[0], top_map[1]], [top_map[0], bot_map[1]]],
                {opacity: 0.6}
            );
        }
    }
    
    updateMap(map,visType,currentSat,year) {
        this.removeCurrentOverlay(map);
        let nextOverlay =  this.imageOverlays[visType][currentSat];
        if (visType == "ntl_intensity") nextOverlay = nextOverlay[year];
        console.log("Incroyable",nextOverlay,this.imageOverlays,year);
        this.currentOverlay = nextOverlay ? nextOverlay.addTo(map) : null;
        updateOpacity(document.getElementById('opacityRange').value);
    }
    
    removeCurrentOverlay(map) {
        if (this.currentOverlay) map.removeLayer(this.currentOverlay);
    }
    
}

function getImgName(color){
    return color == "purple"?"hist_img":"kmeans_img"
}

function updateHist(year){
    var img = document.getElementById("hist_img");
    if (img){
        let path = img.src.split("/")
        let file = path[path.length-1].split("_")
        file[file.length-1] = year + ".png" 
        path[path.length-1] = file.join("_")
        img.src = path.join("/");
    }
}
function updateImage(sat) {
    if(sat == "null"){
        var closeButton = document.getElementsByClassName("leaflet-popup-close-button")[0];
        console.log(closeButton);
        if(closeButton){
            console.log("bruh");
            closeButton.click() ;
            return;
        }
    }
    var img = document.getElementById("hist_img");
    if (img){
        let path = img.src.split("/");
        let file = path[path.length-1].split("_");
        file[2] = sat;
        path[path.length-1] = file.join("_");
        img.src = path.join("/");
        return;
    }
    img = document.getElementById("kmeans_img");
    if (img){
        let path = img.src.split("/");
        path[path.length -3] = sat;
        img.src = path.join("/");
        document.getElementById("kmeans_sat").innerText = sat == "DMSP" ? "DMSP" : sat == "VIIRS" ? "VIIRS" : "none" ;
        console.log(document.getElementById("kmeans_sat") , sat == "DMSP" ? "DMSP" : sat == "VIIRS" ? "VIIRS" : "none" );
        return;
    }
}

