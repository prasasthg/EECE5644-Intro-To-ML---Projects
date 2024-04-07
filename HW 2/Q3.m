clear all; close all; clc;

c_L = vecSpace(0.0001,500,100,"geom");

r = sqrt(rand(1,1));
angle = rand(1,1) * 2 * pi;
x = r .* cos(angle);
y = r .* sin(angle);
trueP = [x; y];

var_X = 0.25;
var_Y = 0.25;
var_I = 0.3;

v_set = [1,2,3,4];

for i = 1:length(v_set)
    m = [];
    LM_pos = landmark_pos(i);
    for j = 1:size(LM_pos,1)
        m = [m; measure(trueP,LM_pos(j,:))];
    end
    [X,Y] = meshgrid(linspace(-2,2,128),linspace(-2,2,128));
    grid_coord = cat(3,X,Y);
    cvalues = MAP_estimation(m,grid_coord,LM_pos,var_X,var_Y);
    plotting(X,Y,c_L,trueP,m,grid_coord,cvalues,LM_pos)
end

function landmarks = landmark_pos(k)
    angles = linspace(0,2*pi,k+1);
    angles = angles(1:k);
    landmarks = [cos(angles)',sin(angles)'];
end

function [measurement] = measure(trueP,lm)
    diff_T = norm(trueP - lm);
    while true
        noise = normrnd(0,0.3);
        measurement = diff_T + noise;
        if measurement >= 0
            break;
        end
    end
end
function post = MAP_estimation(measurements,mh_grid,lndmrk,var_X,var_Y)
    priori = bsxfun(@times,mh_grid,reshape(inv([var_X^2,0; 0,var_Y^2]),[1,1,2,2]));
    priori = priori .* permute(mh_grid,[1,2,4,3]);
    priori = squeeze(priori);
    range_sum = 0;
    for i = 1:length(measurements)
        r_i = measurements(i);
        lm = lndmrk(i,:);
        for j = 1:1
            lm = lndmrk(j,:);
            lm = reshape(lm,[1,1,1,2]); % add an extra dimension
            diff_i = sqrt(sum((mh_grid - lm).^2,4));
            range_sum = range_sum + ((measurements(i) - diff_i).^2 / 0.3^2);
        end
        diff_i = sqrt(sum((mh_grid - lm).^2,3));
        range_sum = range_sum + ((r_i - diff_i).^2) / 0.3^2;
    end
    result = priori + range_sum;
    post = result;
end

function plotting(X,Y,c_L,xy_T,r_meas,gp,cvalues,lm)    
    hold on;
    [C,h] = contour(gp(:,:,1),gp(:,:,2),cvalues(:,:,1),c_L,'LineWidth',1.5,'LineStyle','-');
    colormap(parula(length(c_L)-1));
    hold off;
    fun = @(xy) sum((sqrt(sum((repmat(xy,1,size(r_meas,2)) - landmark_pos(size(r_meas,2))).^2)) - repmat(r_meas,2,1)).^2);
    figure
    hold on;
    axis equal
    unit_circle = viscircles([0,0],1,'Color','k','LineWidth',1.5,'LineStyle','-');
    lm = landmark_pos(length(r_meas));
    lm = squeeze(lm);
    for i = 1:length(r_meas)
        x = lm(i,1);
        y = lm(i,2);
        range_circle = viscircles([x,y],r_meas(i),'Color','r','LineWidth',1.5,'LineStyle','-');
        plot(x,y,'go','MarkerFaceColor','none','MarkerSize',6,'LineWidth',1.5);
    end
    plot(xy_T(1),xy_T(2),'o','MarkerSize',8,'LineWidth',1.5);
    xlabel('X')
    ylabel('Y')
    title(['MAP Estimation for k = ',num2str(length(r_meas))])
    xlim([-2,2])
    ylim([-2,2])
    c = colorbar;
    hold off;
end