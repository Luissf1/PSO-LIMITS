"""
Visualización profesional para resultados de artículo científico
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


class PublicationPlots:
    """Generador de gráficos con estilo para publicación científica"""
    
    def __init__(self, style: str = 'science', dpi: int = 300):
        """
        Args:
            style: Estilo de gráfico ('science', 'default', 'presentation')
            dpi: Resolución para guardar
        """
        self.dpi = dpi
        
        if style == 'science':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.figsize_single = (6, 4)
            self.figsize_double = (10, 6)
            self.font_size = 10
        elif style == 'presentation':
            plt.style.use('seaborn-v0_8-darkgrid')
            self.figsize_single = (8, 6)
            self.figsize_double = (14, 8)
            self.font_size = 14
        else:
            plt.style.use('default')
            self.figsize_single = (6, 4)
            self.figsize_double = (10, 6)
            self.font_size = 10
        
        # Configurar colores (accesibles para daltónicos)
        self.colors = {
            'best': '#2E86AB',   # Azul
            'clerc': '#A23B72',  # Morado
            'dynamic': '#F18F01', # Naranja
            'fraction': '#C73E1D', # Rojo
            'fixed': '#6A994E'    # Verde
        }
        
        sns.set_palette(list(self.colors.values()))
        
    def convergence_comparison(self, results_dict: Dict[str, List[float]],
                              save_path: Optional[Path] = None):
        """
        Gráfico de convergencia comparativo
        """
        fig, ax = plt.subplots(figsize=self.figsize_double)
        
        for strategy, conv_history in results_dict.items():
            # Calcular media y desviación
            conv_array = np.array(conv_history)
            mean_conv = np.mean(conv_array, axis=0)
            std_conv = np.std(conv_array, axis=0)
            
            iterations = range(1, len(mean_conv) + 1)
            
            # Dibujar banda de desviación
            ax.fill_between(iterations, 
                           mean_conv - std_conv, 
                           mean_conv + std_conv,
                           alpha=0.2, 
                           color=self.colors.get(strategy.split('_')[0], 'gray'))
            
            # Dibujar línea media
            ax.plot(iterations, mean_conv, 
                   label=strategy, 
                   linewidth=2,
                   color=self.colors.get(strategy.split('_')[0], 'gray'))
        
        ax.set_xlabel('Iteración', fontsize=self.font_size, fontweight='bold')
        ax.set_ylabel('Fitness (menor es mejor)', fontsize=self.font_size, fontweight='bold')
        ax.set_title('Comparación de Estrategias de Límite de Velocidad',
                    fontsize=self.font_size+2, fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.font_size-1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, len(mean_conv)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def fitness_boxplot(self, results_df: pd.DataFrame,
                       save_path: Optional[Path] = None):
        """
        Boxplot comparativo de fitness
        """
        fig, ax = plt.subplots(figsize=self.figsize_double)
        
        # Ordenar por mediana
        order = results_df.groupby('strategy')['fitness'].median().sort_values().index
        
        # Crear boxplot
        bp = results_df.boxplot(column='fitness', by='strategy', 
                                ax=ax, grid=False, rot=45,
                                order=order, patch_artist=True)
        
        # Personalizar colores
        for i, box in enumerate(bp['boxes']):
            strategy = order[i]
            color = self.colors.get(strategy.split('_')[0], '#CCCCCC')
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        ax.set_xlabel('Estrategia de Límite', fontsize=self.font_size, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=self.font_size, fontweight='bold')
        ax.set_title('Distribución del Fitness por Estrategia',
                    fontsize=self.font_size+2, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Añadir línea para mejor fitness teórico
        min_fitness = results_df['fitness'].min()
        ax.axhline(y=min_fitness, color='green', linestyle='--', 
                  alpha=0.7, label=f'Mejor fitness: {min_fitness:.4f}')
        
        ax.legend(fontsize=self.font_size-1)
        
        plt.suptitle('')  # Quitar título automático
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def metrics_radar(self, strategy_metrics: Dict[str, Dict],
                     save_path: Optional[Path] = None):
        """
        Gráfico radar para comparación multi-métrica
        """
        # Preparar datos
        strategies = list(strategy_metrics.keys())
        metrics = list(strategy_metrics[strategies[0]].keys())
        num_metrics = len(metrics)
        
        # Ángulos para el radar
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        for strategy in strategies:
            values = [strategy_metrics[strategy][m] for m in metrics]
            values += values[:1]  # Cerrar
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=strategy, color=self.colors.get(strategy.split('_')[0], 'gray'))
            ax.fill(angles, values, alpha=0.15)
        
        # Configurar etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=self.font_size-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                          fontsize=self.font_size-2)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                 fontsize=self.font_size-1)
        ax.set_title('Comparación de Métricas Normalizadas',
                    fontsize=self.font_size+2, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def statistical_analysis(self, results_df: pd.DataFrame,
                            save_path: Optional[Path] = None):
        """
        Análisis estadístico con intervalos de confianza
        """
        fig, ax = plt.subplots(figsize=self.figsize_double)
        
        # Calcular estadísticas
        stats_df = results_df.groupby('strategy')['fitness'].agg([
            'mean', 'std', 'count', 'sem'
        ]).reset_index()
        
        # Ordenar por media
        stats_df = stats_df.sort_values('mean')
        
        # Intervalos de confianza del 95%
        stats_df['ci'] = stats_df['sem'] * 1.96
        
        # Crear barras con error
        x_pos = np.arange(len(stats_df))
        bars = ax.bar(x_pos, stats_df['mean'], yerr=stats_df['ci'],
                     capsize=5, alpha=0.7, 
                     color=[self.colors.get(s.split('_')[0], '#CCCCCC') 
                           for s in stats_df['strategy']])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['strategy'], rotation=45, ha='right')
        ax.set_ylabel('Fitness (menor es mejor)', fontsize=self.font_size, fontweight='bold')
        ax.set_title('Comparación Estadística - Media e IC 95%',
                    fontsize=self.font_size+2, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Añadir etiquetas de valores
        for i, (_, row) in enumerate(stats_df.iterrows()):
            ax.text(i, row['mean'] + row['ci'] + 0.01, 
                   f'{row["mean"]:.3f}', ha='center', fontsize=self.font_size-1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()