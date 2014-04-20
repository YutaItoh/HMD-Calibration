%% POINTER: Create a "pointer".
% A "pointer" implemention based on closures
% Modifired from <http://d.hatena.ne.jp/saitodevel01/20101210/1291978535>
%
% P = POINTER(VALUE) create a pointer P which contains VALUE.
% P.get() returns VALUE
% P.set(X) sets a new value X
%
% Example:
% p1 = Pointer(1);
% p2 = p1
% p1.set(123);
% p1.get() == p1.get() % is true

% Copyright (c) Yuta Itoh 2014

function self = Pointer(pointee)
  pointer = pointee;
  self = struct;
  function retval = get()
    retval = pointer;
  end
  function retval = set(new_pointee)
    pointer = new_pointee;
    retval = self;
  end
  self.get = @get;
  self.set = @set;
end