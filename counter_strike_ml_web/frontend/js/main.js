// ============================================================================
// COUNTER STRIKE ML PREDICTOR - FRONTEND JAVASCRIPT COMPLETO
// ============================================================================

class CSMLPredictor {
   constructor() {
       this.apiUrl = 'http://localhost:5000/api';
       this.charts = {};
       this.initializeEventListeners();
       this.showNotification('🎮 Counter Strike ML Predictor cargado', 'info');
   }

   initializeEventListeners() {
       // Botón de predicción de regresión
       const regressionBtn = document.getElementById('predict-regression');
       if (regressionBtn) {
           regressionBtn.addEventListener('click', () => this.handleRegressionPrediction());
       }

       // Botón de predicción de clasificación
       const classificationBtn = document.getElementById('predict-classification');
       if (classificationBtn) {
           classificationBtn.addEventListener('click', () => this.handleClassificationPrediction());
       }

       // Listeners para cambios en el formulario
       this.setupFormListeners();
   }

   setupFormListeners() {
       // Actualizar valores de sliders
       const equipmentSlider = document.getElementById('equipment-value');
       const equipmentDisplay = document.getElementById('equipment-display');
       
       if (equipmentSlider && equipmentDisplay) {
           equipmentSlider.addEventListener('input', (e) => {
               const value = parseInt(e.target.value);
               equipmentDisplay.textContent = '$' + value.toLocaleString();
           });
       }
       
       const teamSlider = document.getElementById('team-equipment');
       const teamDisplay = document.getElementById('team-equipment-display');
       
       if (teamSlider && teamDisplay) {
           teamSlider.addEventListener('input', (e) => {
               const value = parseInt(e.target.value);
               teamDisplay.textContent = '$' + value.toLocaleString();
           });
       }
   }

   collectFormData() {
       const formData = {
           map: document.getElementById('map-select')?.value || 'de_dust2',
           equipment: parseInt(document.getElementById('equipment-value')?.value || 3000),
           teamEquipment: parseInt(document.getElementById('team-equipment')?.value || 15000),
           kills: 5,
           headshots: 2,
           roundKills: 1,
           weapon: this.getSelectedWeapon()
       };

       console.log('📝 Datos del formulario:', formData);
       return formData;
   }

   getSelectedWeapon() {
       const weaponRadios = document.querySelectorAll('input[name="weapon"]');
       for (const radio of weaponRadios) {
           if (radio.checked) {
               return radio.value;
           }
       }
       return 'rifle'; // Valor por defecto
   }

   async handleRegressionPrediction() {
       console.log('🌳 Iniciando predicción de regresión...');
       
       try {
           const formData = this.collectFormData();
           const result = await this.makePrediction('/predict/regression', formData);
           
           if (result.error) {
               this.showError('Regresión', result.error);
               return;
           }

           this.displayRegressionResult(result);
           this.showNotification('✅ Predicción de tiempo completada', 'success');
           
       } catch (error) {
           console.error('❌ Error en predicción de regresión:', error);
           this.showError('Regresión', 'Error de conexión con el servidor');
       }
   }

   async handleClassificationPrediction() {
       console.log('🚀 Iniciando predicción de clasificación...');
       
       try {
           const formData = this.collectFormData();
           const result = await this.makePrediction('/predict/classification', formData);
           
           if (result.error) {
               this.showError('Clasificación', result.error);
               return;
           }

           this.displayClassificationResult(result);
           this.showNotification('✅ Predicción de supervivencia completada', 'success');
           
       } catch (error) {
           console.error('❌ Error en predicción de clasificación:', error);
           this.showError('Clasificación', 'Error de conexión con el servidor');
       }
   }

   async makePrediction(endpoint, data) {
       const response = await fetch(this.apiUrl + endpoint, {
           method: 'POST',
           headers: {
               'Content-Type': 'application/json',
           },
           body: JSON.stringify(data)
       });

       if (!response.ok) {
           const errorData = await response.text();
           throw new Error(`HTTP ${response.status}: ${response.statusText}\n${errorData}`);
       }

       return await response.json();
   }

   displayRegressionResult(result) {
       console.log('📊 Resultado de regresión:', result);

       // Obtener valores seguros
       const prediction = this.safeNumber(result.prediction, 0);
       const confidence = this.safeNumber(result.confidence, 0);
       
       // Mostrar tiempo de supervivencia
       const timeElement = document.getElementById('survival-time');
       const confidenceElement = document.getElementById('regression-confidence');
       
       if (timeElement) {
           if (prediction > 0) {
               timeElement.textContent = `${prediction.toFixed(1)} segundos`;
               timeElement.style.color = this.getTimeColor(prediction);
           } else {
               timeElement.textContent = 'No determinado';
               timeElement.style.color = '#ff6b6b';
           }
       }
       
       if (confidenceElement) {
           confidenceElement.textContent = `Confianza: ${(confidence * 100).toFixed(1)}%`;
           confidenceElement.style.color = this.getConfidenceColor(confidence);
       }

       // Mostrar sección de resultados
       const resultsSection = document.getElementById('results-section');
       if (resultsSection) {
           resultsSection.style.display = 'block';
           resultsSection.scrollIntoView({ behavior: 'smooth' });
       }

       // Crear gráfico de tiempo
       this.createTimeChart(prediction, confidence);
   }

   displayClassificationResult(result) {
       console.log('📊 Resultado de clasificación:', result);

       // Obtener valores seguros
       const prediction = this.safeNumber(result.prediction, 0);
       const probabilityLow = this.safeNumber(result.probability_low, 0.5);
       const probabilityHigh = this.safeNumber(result.probability_high, 0.5);
       const confidence = this.safeNumber(result.confidence, 0.5);

       // Determinar el resultado
       const survivalLabel = prediction === 1 ? 'Alta Supervivencia' : 'Baja Supervivencia';
       const survivalProbability = prediction === 1 ? probabilityHigh : probabilityLow;
       
       // Mostrar clasificación
       const classElement = document.getElementById('survival-class');
       const probElement = document.getElementById('classification-confidence');
       
       if (classElement) {
           classElement.textContent = survivalLabel;
           classElement.style.color = prediction === 1 ? '#4ecdc4' : '#ff6b6b';
       }
       
       if (probElement) {
           probElement.textContent = `Probabilidad: ${(survivalProbability * 100).toFixed(1)}%`;
           probElement.style.color = this.getConfidenceColor(survivalProbability);
       }

       // Mostrar sección de resultados
       const resultsSection = document.getElementById('results-section');
       if (resultsSection) {
           resultsSection.style.display = 'block';
           resultsSection.scrollIntoView({ behavior: 'smooth' });
       }

       // Crear gráfico de clasificación
       this.createClassificationChart(probabilityLow, probabilityHigh, prediction);

       // Mostrar análisis detallado
       this.showDetailedAnalysis(prediction, probabilityLow, probabilityHigh);
   }

   interpretSurvivalTime(timeSeconds) {
       if (timeSeconds <= 0) return 'Tiempo no determinado';
       if (timeSeconds < 15) return '⚡ Eliminación muy rápida - Alto riesgo';
       if (timeSeconds < 30) return '🔥 Eliminación rápida - Riesgo elevado';
       if (timeSeconds < 60) return '⚖️ Supervivencia moderada - Riesgo medio';
       if (timeSeconds < 90) return '💪 Buena supervivencia - Situación favorable';
       return '🏆 Excelente supervivencia - Situación muy favorable';
   }

   interpretClassification(prediction, probability) {
       const percentage = (probability * 100).toFixed(1);
       
       if (prediction === 1) {
           if (probability > 0.8) {
               return `🎯 Muy alta probabilidad de sobrevivir (${percentage}%) - Configuración excelente`;
           } else if (probability > 0.6) {
               return `✅ Buena probabilidad de sobrevivir (${percentage}%) - Configuración favorable`;
           } else {
               return `⚖️ Probabilidad moderada de sobrevivir (${percentage}%) - Situación incierta`;
           }
       } else {
           if (probability > 0.8) {
               return `⚠️ Muy alta probabilidad de eliminación (${percentage}%) - Alto riesgo`;
           } else if (probability > 0.6) {
               return `🚨 Alta probabilidad de eliminación (${percentage}%) - Riesgo considerable`;
           } else {
               return `⚖️ Probabilidad moderada de eliminación (${percentage}%) - Situación incierta`;
           }
       }
   }

   showDetailedAnalysis(prediction, probLow, probHigh) {
       const analysisElement = document.getElementById('detailed-analysis');
       if (!analysisElement) return;

       const survivalChance = probHigh * 100;
       const eliminationChance = probLow * 100;
       
       let analysis = `<h4>📊 Análisis Detallado:</h4>`;
       analysis += `<p><strong>Probabilidad de Supervivencia:</strong> ${survivalChance.toFixed(1)}%</p>`;
       analysis += `<p><strong>Probabilidad de Eliminación:</strong> ${eliminationChance.toFixed(1)}%</p>`;
       
       if (prediction === 1) {
           analysis += `<p class="prediction-positive">✅ <strong>Predicción:</strong> Es probable que sobrevivas esta ronda</p>`;
           if (survivalChance > 75) {
               analysis += `<p class="recommendation">💡 <strong>Recomendación:</strong> Mantén tu estrategia actual, tienes ventaja</p>`;
           } else {
               analysis += `<p class="recommendation">💡 <strong>Recomendación:</strong> Juega con precaución para mantener la ventaja</p>`;
           }
       } else {
           analysis += `<p class="prediction-negative">⚠️ <strong>Predicción:</strong> Alto riesgo de eliminación en esta ronda</p>`;
           if (eliminationChance > 75) {
               analysis += `<p class="recommendation">💡 <strong>Recomendación:</strong> Considera cambiar estrategia o buscar mejor posición</p>`;
           } else {
               analysis += `<p class="recommendation">💡 <strong>Recomendación:</strong> Juega defensivamente y busca ventaja</p>`;
           }
       }
       
       analysisElement.innerHTML = analysis;
   }

   createTimeChart(prediction, confidence) {
       const canvas = document.getElementById('prediction-chart');
       if (!canvas) {
           console.log('📊 Canvas prediction-chart no encontrado');
           return;
       }

       const ctx = canvas.getContext('2d');
       
       // Limpiar canvas
       ctx.clearRect(0, 0, canvas.width, canvas.height);
       
       // Configurar gráfico de barras
       const maxTime = 120; // Máximo tiempo de ronda en CS
       const barWidth = canvas.width * 0.6;
       const barHeight = 40;
       const barX = (canvas.width - barWidth) / 2;
       const barY = (canvas.height - barHeight) / 2;
       
       // Dibujar fondo de la barra
       ctx.fillStyle = '#333';
       ctx.fillRect(barX, barY, barWidth, barHeight);
       
       // Dibujar barra de tiempo
       const timeRatio = Math.min(prediction / maxTime, 1);
       const timeBarWidth = barWidth * timeRatio;
       
       ctx.fillStyle = this.getTimeColor(prediction);
       ctx.fillRect(barX, barY, timeBarWidth, barHeight);
       
       // Texto
       ctx.fillStyle = '#fff';
       ctx.font = '14px Arial';
       ctx.textAlign = 'center';
       ctx.fillText(`${prediction.toFixed(1)}s`, canvas.width / 2, barY + barHeight + 20);
       ctx.fillText(`Confianza: ${(confidence * 100).toFixed(1)}%`, canvas.width / 2, barY + barHeight + 40);
   }

   createClassificationChart(probLow, probHigh, prediction) {
       const canvas = document.getElementById('classification-chart');
       if (!canvas) {
           console.log('📊 Canvas classification-chart no encontrado');
           return;
       }

       const ctx = canvas.getContext('2d');
       
       // Limpiar canvas
       ctx.clearRect(0, 0, canvas.width, canvas.height);
       
       // Configurar gráfico circular
       const centerX = canvas.width / 2;
       const centerY = canvas.height / 2;
       const radius = Math.min(centerX, centerY) - 20;
       
       // Dibujar gráfico circular
       const startAngle = -Math.PI / 2;
       const lowAngle = startAngle + (probLow * 2 * Math.PI);
       
       // Segmento de baja supervivencia
       ctx.beginPath();
       ctx.moveTo(centerX, centerY);
       ctx.arc(centerX, centerY, radius, startAngle, lowAngle);
       ctx.fillStyle = '#ff6b6b';
       ctx.fill();
       
       // Segmento de alta supervivencia
       ctx.beginPath();
       ctx.moveTo(centerX, centerY);
       ctx.arc(centerX, centerY, radius, lowAngle, startAngle + 2 * Math.PI);
       ctx.fillStyle = '#4ecdc4';
       ctx.fill();
       
       // Círculo interior
       ctx.beginPath();
       ctx.arc(centerX, centerY, radius * 0.5, 0, 2 * Math.PI);
       ctx.fillStyle = '#2c3e50';
       ctx.fill();
       
       // Texto central
       ctx.fillStyle = '#fff';
       ctx.font = 'bold 16px Arial';
       ctx.textAlign = 'center';
       const percentage = (prediction === 1 ? probHigh : probLow) * 100;
       ctx.fillText(`${percentage.toFixed(1)}%`, centerX, centerY);
       
       // Leyenda
       ctx.font = '12px Arial';
       ctx.fillStyle = '#ff6b6b';
       ctx.fillText('Eliminación', centerX, centerY + radius + 20);
       ctx.fillStyle = '#4ecdc4';
       ctx.fillText('Supervivencia', centerX, centerY + radius + 35);
   }

   getTimeColor(time) {
       if (time < 15) return '#ff6b6b';      // Rojo - muy poco tiempo
       if (time < 30) return '#ffa500';      // Naranja - poco tiempo
       if (time < 60) return '#ffeb3b';      // Amarillo - tiempo moderado
       if (time < 90) return '#8bc34a';      // Verde claro - buen tiempo
       return '#4caf50';                     // Verde - excelente tiempo
   }

   getConfidenceColor(confidence) {
       if (confidence < 0.3) return '#ff6b6b';      // Baja confianza
       if (confidence < 0.6) return '#ffa500';      // Confianza media
       if (confidence < 0.8) return '#ffeb3b';      // Buena confianza
       return '#4caf50';                            // Alta confianza
   }

   safeNumber(value, defaultValue = 0) {
       if (value === null || value === undefined || isNaN(value)) {
           return defaultValue;
       }
       return Number(value);
   }

   showError(type, message) {
       const timeElement = document.getElementById('survival-time');
       const classElement = document.getElementById('survival-class');
       
       if (type === 'Regresión' && timeElement) {
           timeElement.textContent = 'Error';
           timeElement.style.color = '#ff6b6b';
       }
       
       if (type === 'Clasificación' && classElement) {
           classElement.textContent = 'Error';
           classElement.style.color = '#ff6b6b';
       }
       
       this.showNotification(`❌ Error en ${type}: ${message}`, 'error');
   }

   showNotification(message, type = 'info') {
       console.log(`📢 Notificación ${type}: ${message}`);
       
       // Crear elemento de notificación si no existe
       let notification = document.getElementById('notification');
       if (!notification) {
           notification = document.createElement('div');
           notification.id = 'notification';
           notification.style.cssText = `
               position: fixed;
               top: 20px;
               right: 20px;
               padding: 15px 20px;
               border-radius: 8px;
               color: white;
               font-weight: bold;
               z-index: 1000;
               max-width: 300px;
               opacity: 0;
               transition: opacity 0.3s ease;
           `;
           document.body.appendChild(notification);
       }
       
       // Configurar estilo según tipo
       const colors = {
           success: '#4caf50',
           error: '#ff6b6b',
           warning: '#ffa500',
           info: '#2196f3'
       };
       
       notification.style.backgroundColor = colors[type] || colors.info;
       notification.textContent = message;
       notification.style.opacity = '1';
       
       // Ocultar después de 4 segundos
       setTimeout(() => {
           notification.style.opacity = '0';
       }, 4000);
   }

   // Método de prueba
   async testPrediction() {
       console.log('🧪 Ejecutando prueba de predicción...');
       
       const testData = {
           map: 'de_dust2',
           equipment: 4000,
           teamEquipment: 16000,
           kills: 3,
           headshots: 2,
           roundKills: 1,
           weapon: 'rifle'
       };
       
       try {
           const regResult = await this.makePrediction('/predict/regression', testData);
           const classResult = await this.makePrediction('/predict/classification', testData);
           
           console.log('✅ Prueba de regresión:', regResult);
           console.log('✅ Prueba de clasificación:', classResult);
           
           this.showNotification('✅ Prueba completada exitosamente', 'success');
       } catch (error) {
           console.error('❌ Error en prueba:', error);
           this.showNotification('❌ Error en prueba de conexión', 'error');
       }
   }
}

// Inicializar cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', () => {
   console.log('🎮 Inicializando Counter Strike ML Predictor...');
   window.CSMLPredictor = new CSMLPredictor();
});

// Función global para testing
function testConnection() {
   if (window.CSMLPredictor) {
       window.CSMLPredictor.testPrediction();
   } else {
       console.error('❌ CSMLPredictor no está inicializado');
   }
}