function bool = callback_test(iter, obj)
    fprintf('Callback called for %d iteration with parameter:\n', iter);
    %disp(cmt.Trainable.loadobj(params));
    bool = true;
    