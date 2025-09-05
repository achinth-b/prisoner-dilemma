from classes import Prisoner, Dilemma
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
import os

def run_simulation(p1: float, p2: float, num_rounds: int = 1000) -> Tuple[float, float]:
    """Run a simulation with given defection probabilities over multiple rounds"""
    prisoner1 = Prisoner(p1)
    prisoner2 = Prisoner(p2)
    dilemma = Dilemma()
    
    total_payoff1 = 0
    total_payoff2 = 0
    
    for _ in range(num_rounds):
        payoff1, payoff2 = dilemma.play_round(prisoner1, prisoner2)
        total_payoff1 += payoff1
        total_payoff2 += payoff2
    
    return total_payoff1 / num_rounds, total_payoff2 / num_rounds

def analyze_probability_space(resolution: int = 50, num_rounds: int = 1000) -> Dict:
    """Analyze the entire probability space to find optimal strategies"""
    dilemma = Dilemma()
    
    # Create probability ranges for defection
    prob_range = np.linspace(0, 1, resolution)
    
    # Store results
    results = {
        'p1_defect': [],
        'p2_defect': [],
        'expected_payoff1': [],
        'expected_payoff2': [],
        'simulated_payoff1': [],
        'simulated_payoff2': [],
        'total_welfare': [],
        'nash_equilibrium_distance': []
    }
    
    print(f"Analyzing {resolution}x{resolution} = {resolution**2} strategy combinations...")
    
    for i, p1 in enumerate(prob_range):
        if i % 10 == 0:
            print(f"Progress: {i}/{resolution} ({100*i/resolution:.1f}%)")
        
        for p2 in prob_range:
            # Calculate expected payoffs analytically
            exp_p1, exp_p2 = dilemma.expected_payoff(p1, p2)
            
            # Run simulation (smaller sample for speed)
            sim_p1, sim_p2 = run_simulation(p1, p2, num_rounds//10)
            
            # Calculate total welfare
            total_welfare = exp_p1 + exp_p2
            
            # Calculate distance from Nash equilibrium (both always defect)
            nash_distance = np.sqrt((p1 - 1)**2 + (p2 - 1)**2)
            
            results['p1_defect'].append(p1)
            results['p2_defect'].append(p2)
            results['expected_payoff1'].append(exp_p1)
            results['expected_payoff2'].append(exp_p2)
            results['simulated_payoff1'].append(sim_p1)
            results['simulated_payoff2'].append(sim_p2)
            results['total_welfare'].append(total_welfare)
            results['nash_equilibrium_distance'].append(nash_distance)
    
    print("Analysis complete!")
    return results

def find_optimal_strategies(results: Dict) -> Dict:
    """Find various optimal strategies based on different criteria"""
    
    # Convert to numpy arrays for easier analysis
    p1_vals = np.array(results['p1_defect'])
    p2_vals = np.array(results['p2_defect'])
    payoff1 = np.array(results['expected_payoff1'])
    payoff2 = np.array(results['expected_payoff2'])
    total_welfare = np.array(results['total_welfare'])
    
    # Find indices for different optimal strategies
    max_welfare_idx = np.argmax(total_welfare)
    max_p1_payoff_idx = np.argmax(payoff1)
    max_p2_payoff_idx = np.argmax(payoff2)
    
    # Find Pareto optimal solutions (maximize minimum payoff)
    min_payoffs = np.minimum(payoff1, payoff2)
    max_min_payoff_idx = np.argmax(min_payoffs)
    
    return {
        'max_welfare': {
            'p1_defect': p1_vals[max_welfare_idx],
            'p2_defect': p2_vals[max_welfare_idx],
            'payoff1': payoff1[max_welfare_idx],
            'payoff2': payoff2[max_welfare_idx],
            'total_welfare': total_welfare[max_welfare_idx]
        },
        'max_p1_payoff': {
            'p1_defect': p1_vals[max_p1_payoff_idx],
            'p2_defect': p2_vals[max_p1_payoff_idx],  # Fixed bug here
            'payoff1': payoff1[max_p1_payoff_idx],
            'payoff2': payoff2[max_p1_payoff_idx]
        },
        'max_p2_payoff': {
            'p1_defect': p1_vals[max_p2_payoff_idx],
            'p2_defect': p2_vals[max_p2_payoff_idx],
            'payoff1': payoff1[max_p2_payoff_idx],
            'payoff2': payoff2[max_p2_payoff_idx]
        },
        'max_min_payoff': {
            'p1_defect': p1_vals[max_min_payoff_idx],
            'p2_defect': p2_vals[max_min_payoff_idx],
            'payoff1': payoff1[max_min_payoff_idx],
            'payoff2': payoff2[max_min_payoff_idx],
            'min_payoff': min_payoffs[max_min_payoff_idx]
        }
    }

def plot_analysis(results: Dict, optimal_strategies: Dict, save_plots: bool = True):
    """Create comprehensive visualizations of the analysis"""
    
    resolution = int(np.sqrt(len(results['p1_defect'])))
    
    # Reshape data for contour plots
    p1_grid = np.array(results['p1_defect']).reshape(resolution, resolution)
    p2_grid = np.array(results['p2_defect']).reshape(resolution, resolution)
    payoff1_grid = np.array(results['expected_payoff1']).reshape(resolution, resolution)
    payoff2_grid = np.array(results['expected_payoff2']).reshape(resolution, resolution)
    welfare_grid = np.array(results['total_welfare']).reshape(resolution, resolution)
    nash_distance_grid = np.array(results['nash_equilibrium_distance']).reshape(resolution, resolution)
    
    # Create output directory if saving plots
    if save_plots:
        os.makedirs('plots', exist_ok=True)
    
    # Create a larger figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Prisoner 1 Expected Payoff
    ax1 = plt.subplot(2, 3, 1)
    contour1 = ax1.contourf(p1_grid, p2_grid, payoff1_grid, levels=20, cmap='viridis')
    ax1.set_xlabel('P1 Probability of Defecting')
    ax1.set_ylabel('P2 Probability of Defecting')
    ax1.set_title('Prisoner 1 Expected Payoff')
    plt.colorbar(contour1, ax=ax1)
    
    # Mark P1's optimal point
    p1_opt = optimal_strategies['max_p1_payoff']
    ax1.plot(p1_opt['p1_defect'], p1_opt['p2_defect'], 'r*', markersize=15, label='P1 Max')
    ax1.legend()
    
    # Plot 2: Prisoner 2 Expected Payoff  
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(p1_grid, p2_grid, payoff2_grid, levels=20, cmap='viridis')
    ax2.set_xlabel('P1 Probability of Defecting')
    ax2.set_ylabel('P2 Probability of Defecting')
    ax2.set_title('Prisoner 2 Expected Payoff')
    plt.colorbar(contour2, ax=ax2)
    
    # Mark P2's optimal point
    p2_opt = optimal_strategies['max_p2_payoff']
    ax2.plot(p2_opt['p1_defect'], p2_opt['p2_defect'], 'r*', markersize=15, label='P2 Max')
    ax2.legend()
    
    # Plot 3: Total Welfare with ALL optimal points
    ax3 = plt.subplot(2, 3, 3)
    contour3 = ax3.contourf(p1_grid, p2_grid, welfare_grid, levels=20, cmap='viridis')
    ax3.set_xlabel('P1 Probability of Defecting')
    ax3.set_ylabel('P2 Probability of Defecting')
    ax3.set_title('Total Welfare (Sum of Payoffs)')
    plt.colorbar(contour3, ax=ax3)
    
    # Mark ALL optimal points
    welfare_opt = optimal_strategies['max_welfare']
    maxmin_opt = optimal_strategies['max_min_payoff']
    
    ax3.plot(welfare_opt['p1_defect'], welfare_opt['p2_defect'], 
             'r*', markersize=15, label='Max Welfare')
    ax3.plot(maxmin_opt['p1_defect'], maxmin_opt['p2_defect'], 
             'b^', markersize=12, label='Max Min Payoff')
    ax3.plot(1, 1, 'ko', markersize=10, label='Nash Equilibrium')  # Always defect
    ax3.legend()
    
    # Plot 4: Nash Distance from Equilibrium
    ax4 = plt.subplot(2, 3, 4)
    contour4 = ax4.contourf(p1_grid, p2_grid, nash_distance_grid, levels=20, cmap='plasma')
    ax4.set_xlabel('P1 Probability of Defecting')
    ax4.set_ylabel('P2 Probability of Defecting')
    ax4.set_title('Distance from Nash Equilibrium')
    plt.colorbar(contour4, ax=ax4)
    
    # Mark Nash equilibrium point
    ax4.plot(1, 1, 'ko', markersize=10, label='Nash Equilibrium')
    ax4.legend()
    
    # Plot 5: Pareto Frontier Analysis
    ax5 = plt.subplot(2, 3, 5)
    
    # Create scatter plot of all payoff combinations
    payoff1_flat = results['expected_payoff1']
    payoff2_flat = results['expected_payoff2']
    
    # Find Pareto efficient points
    pareto_efficient = []
    for i, (p1, p2) in enumerate(zip(payoff1_flat, payoff2_flat)):
        is_pareto = True
        for j, (q1, q2) in enumerate(zip(payoff1_flat, payoff2_flat)):
            if i != j and q1 >= p1 and q2 >= p2 and (q1 > p1 or q2 > p2):
                is_pareto = False
                break
        if is_pareto:
            pareto_efficient.append(i)
    
    # Plot all points and highlight Pareto frontier
    ax5.scatter(payoff1_flat, payoff2_flat, c='lightblue', alpha=0.5, s=1)
    pareto_p1 = [payoff1_flat[i] for i in pareto_efficient]
    pareto_p2 = [payoff2_flat[i] for i in pareto_efficient]
    ax5.scatter(pareto_p1, pareto_p2, c='red', s=20, label='Pareto Frontier')
    
    ax5.set_xlabel('Player 1 Payoff')
    ax5.set_ylabel('Player 2 Payoff')
    ax5.set_title('Payoff Space & Pareto Frontier')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Symmetric Strategy Analysis (AUC-like)
    ax6 = plt.subplot(2, 3, 6)
    prob_range = np.linspace(0, 1, 100)
    symmetric_payoffs = []
    symmetric_welfare = []
    
    dilemma = Dilemma()
    for p in prob_range:
        payoff1, payoff2 = dilemma.expected_payoff(p, p)  # Symmetric case
        symmetric_payoffs.append((payoff1 + payoff2) / 2)  # Average payoff
        symmetric_welfare.append(payoff1 + payoff2)  # Total welfare
    
    ax6.plot(prob_range, symmetric_payoffs, 'b-', linewidth=2, label='Average Payoff')
    ax6.plot(prob_range, symmetric_welfare, 'g-', linewidth=2, label='Total Welfare')
    ax6.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Nash Equilibrium Payoff')
    ax6.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Cooperation Payoff')
    ax6.set_xlabel('Probability of Defecting (Both Players)')
    ax6.set_ylabel('Payoff')
    ax6.set_title('Symmetric Strategy Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("Comprehensive plot saved to: plots/comprehensive_analysis.png")
    
    plt.show()

def analyze_inequalities():
    """Analyze the inequality conditions for optimal cooperation"""
    dilemma = Dilemma()
    
    print("=== INEQUALITY ANALYSIS ===")
    print("\nPayoff Matrix:")
    for outcome, payoffs in dilemma.payoff_matrix.items():
        print(f"{outcome}: {payoffs}")
    
    print("\nFor cooperation to be better than defection:")
    print("Expected payoff from cooperation > Expected payoff from defection")
    print("\nLet p be opponent's probability of defecting:")
    
    # Use actual payoff values from the matrix
    coop_payoff = dilemma.payoff_matrix[('cooperate', 'cooperate')][0]  # 5
    sucker_payoff = dilemma.payoff_matrix[('cooperate', 'defect')][0]   # 0
    tempt_payoff = dilemma.payoff_matrix[('defect', 'cooperate')][0]    # 10
    defect_payoff = dilemma.payoff_matrix[('defect', 'defect')][0]      # 1
    
    print(f"Cooperation expected payoff: {coop_payoff}(1-p) + {sucker_payoff}(p) = {coop_payoff}(1-p)")
    print(f"Defection expected payoff: {tempt_payoff}(1-p) + {defect_payoff}(p) = {tempt_payoff}(1-p) + {defect_payoff}p")
    print(f"\nCooperation is better when: {coop_payoff}(1-p) > {tempt_payoff}(1-p) + {defect_payoff}p")
    print(f"Simplifying: {coop_payoff} - {coop_payoff}p > {tempt_payoff} - {tempt_payoff}p + {defect_payoff}p")
    print(f"{coop_payoff} - {coop_payoff}p > {tempt_payoff} - {tempt_payoff - defect_payoff}p")
    print(f"{coop_payoff - tempt_payoff} > {coop_payoff - tempt_payoff + defect_payoff}p")
    print(f"{coop_payoff - tempt_payoff} > {-(tempt_payoff - coop_payoff - defect_payoff)}p")
    
    threshold = (coop_payoff - tempt_payoff) / (-(tempt_payoff - coop_payoff - defect_payoff))
    print(f"p > {threshold}")
    print(f"\nSince p cannot exceed 1, and {threshold} > 1, pure cooperation is never rational!")
    print("This confirms the prisoner's dilemma structure is maintained.")

def save_results_to_file(results: Dict, optimal_strategies: Dict):
    """Save numerical results to a text file"""
    os.makedirs('results', exist_ok=True)
    
    with open('results/experiment_results.txt', 'w') as f:
        f.write("=== PRISONER'S DILEMMA EXPERIMENT RESULTS ===\n\n")
        
        # Payoff matrix
        dilemma = Dilemma()
        f.write("Payoff Matrix:\n")
        for outcome, payoffs in dilemma.payoff_matrix.items():
            f.write(f"  {outcome}: {payoffs}\n")
        f.write("\n")
        
        # Optimal strategies
        f.write("Optimal Strategies:\n")
        f.write("-" * 50 + "\n")
        for strategy_name, strategy_data in optimal_strategies.items():
            f.write(f"\n{strategy_name.replace('_', ' ').title()}:\n")
            for key, value in strategy_data.items():
                f.write(f"  {key}: {value:.6f}\n")
        
        # Summary statistics
        f.write(f"\nSummary Statistics:\n")
        f.write(f"Total strategy combinations analyzed: {len(results['p1_defect'])}\n")
        f.write(f"Maximum total welfare: {max(results['total_welfare']):.6f}\n")
        f.write(f"Minimum total welfare: {min(results['total_welfare']):.6f}\n")
        f.write(f"Average total welfare: {np.mean(results['total_welfare']):.6f}\n")
    
    print("Results saved to: results/experiment_results.txt")

def plot_inequalities(save_plots: bool = True):
    """Plot the key inequalities that define the prisoner's dilemma"""
    
    if save_plots:
        os.makedirs('plots', exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cooperation vs Defection Inequality
    p_range = np.linspace(0, 1, 100)
    coop_payoff = 5 * (1 - p_range)  # Cooperation expected payoff
    defect_payoff = 10 * (1 - p_range) + p_range  # Defection expected payoff
    
    ax1.plot(p_range, coop_payoff, 'g-', linewidth=3, label='Cooperation Payoff: 5(1-p)')
    ax1.plot(p_range, defect_payoff, 'r-', linewidth=3, label='Defection Payoff: 10(1-p) + p')
    ax1.fill_between(p_range, coop_payoff, defect_payoff, 
                     where=(defect_payoff > coop_payoff), alpha=0.3, color='red',
                     label='Defection Better')
    ax1.set_xlabel('Opponent Probability of Defecting')
    ax1.set_ylabel('Expected Payoff')
    ax1.set_title('Cooperation vs Defection Inequality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 3, 'Defection ALWAYS\nbetter!', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: Payoff Matrix Inequalities
    categories = ['T > R', 'R > P', 'P > S', '2R > T+S']
    values = [10-5, 5-1, 1-0, 10-(10+0)]  # Differences
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Inequality Value (must be > 0)')
    ax2.set_title('Prisoner\'s Dilemma Structure Inequalities')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.3,
                f'{value}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Plot 3: Nash Equilibrium Regions
    resolution = 50
    p1_range = np.linspace(0, 1, resolution)
    p2_range = np.linspace(0, 1, resolution)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Best response for P1 given P2's strategy
    # P1 should always defect (best response is p1=1 regardless of p2)
    best_response_p1 = np.ones_like(P2)
    
    # Best response for P2 given P1's strategy  
    # P2 should always defect (best response is p2=1 regardless of p1)
    best_response_p2 = np.ones_like(P1)
    
    # Nash equilibrium indicator (both players best responding)
    nash_indicator = np.logical_and(P1 >= 0.9, P2 >= 0.9).astype(float)
    
    contour = ax3.contourf(P1, P2, nash_indicator, levels=[0, 0.5, 1], 
                          colors=['lightblue', 'darkred'], alpha=0.7)
    ax3.set_xlabel('P1 Probability of Defecting')
    ax3.set_ylabel('P2 Probability of Defecting') 
    ax3.set_title('Nash Equilibrium Region')
    ax3.plot(1, 1, 'ko', markersize=15, label='Nash Equilibrium')
    ax3.legend()
    
    # Add text annotations
    ax3.text(0.5, 0.5, 'NOT Nash\nEquilibrium', ha='center', va='center', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax3.text(0.95, 0.95, 'Nash\nEquilibrium', ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))
    
    # Plot 4: Welfare Inequality (Individual vs Social Optimum)
    # Show the welfare loss from Nash equilibrium vs social optimum
    strategy_types = ['Social Optimum\n(0,0)', 'Nash Equilibrium\n(1,1)', 'Exploitation\n(1,0)', 'Exploitation\n(0,1)']
    welfare_values = [10, 2, 10, 10]  # Total welfare for each strategy profile
    individual_max = [5, 1, 10, 0]   # Maximum individual payoff in each case
    
    x = np.arange(len(strategy_types))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, welfare_values, width, label='Total Welfare', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, individual_max, width, label='Max Individual Payoff', color='orange', alpha=0.7)
    
    ax4.set_ylabel('Payoff')
    ax4.set_title('Welfare vs Individual Incentives')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategy_types)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height}', ha='center', va='bottom', fontweight='bold')
    
    # Add inequality annotation
    ax4.text(0.5, 8, 'Welfare Loss:\n10 → 2 = 80%!', ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('plots/inequality_analysis.png', dpi=300, bbox_inches='tight')
        print("Inequality analysis saved to: plots/inequality_analysis.png")
    
    plt.show()

def analyze_mathematical_inequalities():
    """Print detailed mathematical analysis of the inequalities"""
    dilemma = Dilemma()
    
    print("\n=== MATHEMATICAL INEQUALITY ANALYSIS ===")
    print("=" * 50)
    
    # Get payoff values
    T = dilemma.payoff_matrix[('defect', 'cooperate')][0]      # 10
    R = dilemma.payoff_matrix[('cooperate', 'cooperate')][0]   # 5  
    P = dilemma.payoff_matrix[('defect', 'defect')][0]         # 1
    S = dilemma.payoff_matrix[('cooperate', 'defect')][0]      # 0
    
    print(f"\nPayoff Matrix Values:")
    print(f"T (Temptation) = {T}")
    print(f"R (Reward) = {R}")  
    print(f"P (Punishment) = {P}")
    print(f"S (Sucker) = {S}")
    
    print(f"\n1. PRISONER'S DILEMMA INEQUALITIES:")
    print(f"   T > R: {T} > {R} = {T > R} ✓")
    print(f"   R > P: {R} > {P} = {R > P} ✓") 
    print(f"   P > S: {P} > {S} = {P > S} ✓")
    print(f"   2R > T+S: {2*R} > {T+S} = {2*R > T+S} {'✓' if 2*R > T+S else '✗'}")
    
    print(f"\n2. COOPERATION vs DEFECTION INEQUALITY:")
    print(f"   Let p = opponent's defection probability")
    print(f"   Cooperation payoff: {R}(1-p) + {S}p = {R}(1-p)")
    print(f"   Defection payoff: {T}(1-p) + {P}p = {T}(1-p) + {P}p")
    print(f"   Cooperation better when: {R}(1-p) > {T}(1-p) + {P}p")
    print(f"   Simplifying: {R-T} > {P-T}p")
    print(f"   Therefore: p > {(R-T)/(P-T)} = {(R-T)/(P-T):.2f}")
    print(f"   Since p ∈ [0,1], cooperation is NEVER better!")
    
    print(f"\n3. WELFARE LOSS INEQUALITY:")
    print(f"   Social optimum welfare: 2R = {2*R}")
    print(f"   Nash equilibrium welfare: 2P = {2*P}")
    print(f"   Welfare loss: {2*R - 2*P} points ({100*(2*R - 2*P)/(2*R):.1f}%)")
    
    print(f"\n4. EXPLOITATION INEQUALITY:")
    print(f"   Exploiter gets: T = {T}")
    print(f"   Victim gets: S = {S}") 
    print(f"   Cooperation gets: R = {R}")
    print(f"   Temptation to exploit: T - R = {T - R} extra points")
    
    return T, R, P, S

def experiment():
    """Main experiment function with high-resolution analysis"""
    print("=== PRISONER'S DILEMMA PROBABILITY ANALYSIS ===\n")
    
    # Run mathematical inequality analysis
    T, R, P, S = analyze_mathematical_inequalities()
    
    # Plot the inequalities
    print("\nGenerating inequality visualizations...")
    plot_inequalities(save_plots=True)
    
    # Run the original analysis
    analyze_inequalities()
    
    print("\n=== HIGH-RESOLUTION PROBABILITY SPACE ANALYSIS ===")
    print("Analyzing defection probability space with 1% precision...")
    print("This will test 101 × 101 = 10,201 strategy combinations...")
    print("Expected runtime: 5-15 minutes depending on your computer")
    
    # Analyze the probability space with high resolution
    results = analyze_probability_space(resolution=101, num_rounds=500)  # Reduced rounds per combo for speed
    
    # Find optimal strategies
    optimal_strategies = find_optimal_strategies(results)
    
    print("\nOptimal Strategies Found:")
    print("-" * 50)
    
    for strategy_name, strategy_data in optimal_strategies.items():
        print(f"\n{strategy_name.replace('_', ' ').title()}:")
        for key, value in strategy_data.items():
            print(f"  {key}: {value:.4f}")
    
    # Save results to file
    save_results_to_file(results, optimal_strategies)
    
    # Create and save visualizations
    plot_analysis(results, optimal_strategies, save_plots=True)
    
    print("\n=== KEY INSIGHTS ===")
    print("1. Nash equilibrium: both always defect (p=1, p=1)")
    print("2. Social optimum: both always cooperate (p=0, p=0)") 
    print("3. The dilemma: individual rationality leads to collective irrationality")
    print("4. ALL inequalities favor defection - cooperation never pays!")
    print("5. Welfare loss from Nash play: 80% of potential welfare destroyed")
    
    print(f"\n=== COMPUTATIONAL SUMMARY ===")
    print(f"Strategy combinations tested: {len(results['p1_defect']):,}")
    print(f"Total simulated games: {len(results['p1_defect']) * 500:,}")
    print(f"Probability precision: 1% (0.01 increments)")

def main():
    try:
        experiment()
    except ImportError as e:
        print("Error: Required packages not found.")
        print("Please install with: pip install matplotlib numpy")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()