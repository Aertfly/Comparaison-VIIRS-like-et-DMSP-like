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
        this.setListener();
        this.map = map;
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
        var handler = this
        $('#year').on('input', function() {
            if(handler.getType()!="kmeans")handler.updateMap($(this).val());
            let span = document.getElementById("yearValue");
            span.innerHTML = "";
            span.appendChild(document.createTextNode($(this).val()));
        });
        
        var checkboxes = document.getElementsByClassName('leaflet-control-layers-selector');
        for (var i = 0; i < checkboxes.length; i++){
            checkboxes[i].addEventListener('change', function() {
                let innerText = this.nextSibling.textContent.trim();
                if (this.checked) {
                    handler.notifyCheck(innerText);
                    const purpleMarker = document.getElementsByClassName("awesome-marker-icon-purple awesome-marker leaflet-zoom-animated leaflet-interactive")
                    console.log(purpleMarker,"ding");
                    for (let pm of purpleMarker){
                        pm.addEventListener("click",()=>{
                            console.log("DINGZ")
                        });
                    }
                }else{
                    handler.notifyUncheck(innerText);
                }
            });
        }
        
        var sat =  document.getElementsByName("sat")
        for(const radio of sat){
            radio.addEventListener('change',()=>{
                if (radio.checked){
                    handler.updateSat (radio.value === "null" ? null :radio.value)
                }
            })
        }
        
        var typeVis =  document.getElementsByName("typeVis")
        for(const radio of typeVis){
            radio.addEventListener('change',()=>{
                if (radio.checked){
                    handler.updateTypeVis (radio.value === "null" ? null :radio.value)
                }
            });
        }
    }
    update(){
        this.updateMap($('#year').val())
    }
    
    notifyCheck(country){
        this.currentlyChecked.push(country);
        this.update();
    }
    
    notifyUncheck(country){
        let del = this.currentlyChecked.splice(this.currentlyChecked.indexOf(country),1);
        if (this.overlays[del])this.overlays[del].removeCurrentOverlay(this.map);
    }
    
    
    updateTypeVis(typeVis){
        this.currentType = typeVis;
        typeVis==null?this.removeFromMap():this.update();
    }
    
    getType(){
        return this.currentType}
        
        updateSat(sat){
            this.currentSat = sat;
            console.log("Nouveau sat",this.currentSat , sat)
            sat==null?this.removeFromMap():this.update();
        }
        
        
    }
    
    class overlay{
        imageOverlays = {"ntl_intensity":{},"kmeans":{}};
        currentOverlay = null;
        
        constructor(first_year,last_year,country,clusters,top_map,bot_map) {
            let path ="";
            
            const sats = ["DMSP", "VIIRS"];
            for (let sat of sats) {
                path = "../analysis/" + country;
                let temp = {}
                for (let i = first_year; i < last_year+1; i++) {
                    let n = "/ntl_intensity_" + i + "_" + sat + ".png";
                    temp[i] = L.imageOverlay(path + '/ntl_intensity/' + sat + "/" + n,
                        [[bot_map[0], top_map[1]], [top_map[0], bot_map[1]]],
                        {opacity: 0.6}
                    );  
                }
                this.imageOverlays["ntl_intensity"][sat] = temp;
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
            if (visType == "ntl_intensity") nextOverlay = nextOverlay[year]
            console.log("Incroyable",nextOverlay,this.imageOverlays,year)
            this.currentOverlay = nextOverlay ? nextOverlay.addTo(map) : null;
            updateOpacity(document.getElementById('opacityRange').value);
        }
        
        removeCurrentOverlay(map) {
            if (this.currentOverlay) map.removeLayer(this.currentOverlay);
        }
        
    }

